from ..queues import AllQueues
from ..experiments import Experiment, ImagingConfig
from ..messages import Message
from ..datatypes import AcquisitionData, SegmentationData, CameraPattern
import logging
from typing import NamedTuple, Union
from uuid import UUID, uuid4
from collections import defaultdict
from pathlib import Path
from h5py import File
import numpy as np
from natsort import natsorted

logger = logging.getLogger(__name__)


AcquiredImageRequest = NamedTuple("AcquiredImageRequest", [("id", UUID), ("needs_raw", bool), ("needs_seg", bool)])
ROI = NamedTuple("ROI", [("x_offset", int), ("y_offset", int), ("width", int), ("height", int)])
CameraProperties = NamedTuple("CameraProperties", [("roi", ROI), ("pixel_size_um", float)])

# class RequestMethod(Message):
#     message = "request_method"
#
#     def __init__(self, experiment: Experiment):
#         self.experiment = experiment
#
#
# class PatternRequirements(Message):
#     message = "pattern_requirements"
#
#     def __init__(self, requirements: list[AcquiredImageRequest]):
#         self.requirements = requirements


class RequestPattern(Message):
    """
    Used by manager to inform pattern process of upcoming pattern generation
    Pattern process will use this information to collect and store incoming
    raw and segmented data
    """

    message = "request_pattern"

    def __init__(self, t, experiment_name: str, requirements: list[AcquiredImageRequest]):

        self.t = t
        self.experiment_name = experiment_name
        self.requirements = requirements


class DataDock:

    def __init__(self, t, requirements: list[AcquiredImageRequest]):

        self.t = t
        self.requirements = requirements
        self.data = defaultdict(dict)

        for channel_id, needs_raw, needs_seg in requirements:
            if needs_raw:
                self.data[channel_id]["raw"] = None

            if needs_seg:
                self.data[channel_id]["seg"] = None

        self.complete = self.check_complete()

    def add_raw(self, data: AcquisitionData):

        channel_id = data.channel_id
        # ensure channel id was expected
        assert channel_id in self.data, "unexpected data passed to pattern module"

        # ensure raw data was expected and not already passed
        assert "raw" in self.data[channel_id], "raw data being passed, but not expected"
        assert self.data[channel_id]["raw"] is None, f"expected none, found {self.data[channel_id]['raw']}"

        self.data[channel_id]["raw"] = data

    def add_seg(self, data: SegmentationData):

        channel_id = data.channel_id
        # ensure channel id was expected
        assert channel_id in self.data, "unexpected data passed to pattern module"

        # ensure raw data was expected and not already passed
        assert "seg" in self.data[channel_id], "seg data being passed, but not expected"
        assert self.data[channel_id]["seg"] is None, f"expected none, found {self.data[channel_id]['seg']}"

        self.data[channel_id]["seg"] = data

    def get_awaiting(self):

        awaiting = []

        for channel in self.data:
            for img in self.data[channel]:
                if self.data[channel][img] is None:
                    awaiting.append((channel, img))

        return awaiting

    def check_complete(self):

        return len(self.get_awaiting()) == 0


class PatternModel:

    name = "base"

    def __init__(self, experiment_name, camera_properties: CameraProperties, **kwargs):
        self.experiment = experiment_name
        self.camera_properties = camera_properties
        self.pixel_size_um = camera_properties.pixel_size_um
        self.pattern_shape = (camera_properties.roi.height, camera_properties.roi.width)
        self.binning = 1

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:

        binning = experiment.stimulation.binning

        self.update_binning(binning)

        return []

    def get_meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.pattern_shape

        y_range = np.arange(h) * self.pixel_size_um
        x_range = np.arange(w) * self.pixel_size_um

        xx, yy = np.meshgrid(x_range, y_range)

        return xx, yy

    def generate(self, data_dock: DataDock) -> np.ndarray:
        raise NotImplementedError

    def update_binning(self, binning: int):

        binning_rescale = binning / self.binning

        self.pixel_size_um = self.pixel_size_um * binning_rescale
        self.pattern_shape = (self.pattern_shape[0] // binning_rescale, self.pattern_shape[1] // binning_rescale)

        logger.info(f"model {self.name} updated pixel size (um) to {self.pixel_size_um}")
        logger.info(f"model {self.name} updated pattern_shape to {self.pattern_shape}")

        self.binning = binning


class PatternModelReturnsSLM(PatternModel):

    pass


class PatternReview(PatternModelReturnsSLM):

    name = "pattern_review"

    def __init__(self, experiment_name, camera_properties, h5fp: Union[str, Path] = None, channel="545", **kwargs):
        super().__init__(experiment_name, camera_properties)

        if h5fp is None:
            raise ValueError("pattern_review model requires specifying h5fp (h5 filepath)")

        self.fp = Path(h5fp)
        self.channel_name = f"channel_{channel}"

        with File(str(self.fp), "r") as f:

            keys = list(f.keys())

        self.keys = natsorted(keys)

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        return []

    def generate(self, data_dock: DataDock) -> np.ndarray:

        with File(str(self.fp), "r") as f:

            while len(self.keys) > 0:

                key = self.keys.pop(0)

                if self.channel_name in f[key]:

                    return np.array(f[key]["stim_aq"]["dmd"])

        return np.array([])


class CirclePattern(PatternModel):
    """
    Projects a circle of radius rad in the center of the camera
    """

    name = "circle"

    def __init__(self, experiment_name, camera_properties, rad=1, **kwargs):
        super().__init__(experiment_name, camera_properties)

        self.rad = rad

    def initialize(self, experiment):

        super().initialize(experiment)

        return []

    def generate(self, data_dock: DataDock):

        h, w = self.pattern_shape
        center_x = w / 2.
        center_y = h / 2.

        xx, yy = self.get_meshgrid()

        pattern = np.ones((h, w), dtype=np.float16)

        return (pattern * (((xx - center_x)**2 + (yy - center_y)**2) < (self.rad**2))).astype(np.float16)


class BarPatternBase(PatternModel):
    def __new__(cls, *args, **kwargs):
        if cls is BarPatternBase:  # Check if the base class is being instantiated
            if kwargs.get("bar_speed") != 0:
                return super().__new__(BarPattern)
            else:
                return super().__new__(StationaryBarPattern)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        print(f"Initializing {self.__class__.__name__}")


class StationaryBarPattern(BarPatternBase):
    """
    moves a bar along the y-axis
    """

    name = "bar (stationary)"

    def __init__(self, experiment_name, camera_properties, duty_cycle=0.2, bar_speed=0, period=30, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        """
        super().__init__(experiment_name, camera_properties)

        self.duty_cycle = duty_cycle
        self.bar_speed = 0
        self.period_space = period    # in um
        self.period_time = 0    # in minutes

    def initialize(self, experiment):
        super().initialize(experiment)

        return []

    def generate(self, data_dock: DataDock):

        xx, yy = self.get_meshgrid()

        is_on = ((yy / self.period_space) % 1.0) < self.duty_cycle

        return is_on.astype(np.float16)


class BarPattern(BarPatternBase):
    """
    moves a bar along the y-axis
    """

    name = "bar"

    def __init__(self, experiment_name, camera_properties, duty_cycle=0.2, bar_speed=1, period=30, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        """
        super().__init__(experiment_name, camera_properties)

        self.duty_cycle = duty_cycle
        self.bar_speed = bar_speed
        self.period_space = period    # in um
        self.period_time = period / bar_speed    # in minutes

    def initialize(self, experiment):
        super().initialize(experiment)

        return []

    def generate(self, data_dock: DataDock):

        t = data_dock.t / 60

        xx, yy = self.get_meshgrid()

        is_on = ((t - (yy / self.bar_speed)) % self.period_time) < self.duty_cycle*self.period_time

        return is_on.astype(np.float16)


# class PatternDataDock:
#
#     def
#

class PatternProcess:

    known_models = {
        "circle": CirclePattern,
        "bar": BarPatternBase,
        "pattern_review": PatternReview,

    }

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_pattern
        self.manager = aq.pattern_to_manager
        self.slm = aq.pattern_to_slm

        self.from_seg = aq.seg_to_pattern
        self.from_raw = aq.outbox_to_pattern

        self.camera_properties = None
        self.initialized = False

        self.models = {}
        self.docks = {}

    def initialize(self, camera_properties: CameraProperties):
        self.camera_properties = camera_properties

        self.initialized = True

    def request_model(self, experiment: Experiment) -> list[AcquiredImageRequest]:

        method_name = experiment.pattern.method_name

        model_class: type = self.known_models.get(method_name)

        assert model_class is not None, f"method {method_name} is not a registered model"
        assert issubclass(model_class, PatternModel), f"{method_name} is not a PatternModel"

        experiment_name = experiment.experiment_name
        method_kwargs = experiment.pattern.kwargs

        model = model_class(experiment_name, self.camera_properties, **method_kwargs)

        self.models[experiment_name] = model

        return model.initialize(experiment)

    def register_model(self, model: type):

        assert issubclass(model, PatternModel), "model must be a subclass of PatternModel"

        model_name = model.name

        if model_name in self.known_models:
            logging.warning(f"overwriting known model {model_name}")

        self.known_models[model_name] = model

    def run_model(self, experiment_name):

        data_dock = self.docks.pop(experiment_name)

        model = self.models.get(experiment_name, None)

        assert isinstance(model, PatternModel), f"self.models[{'experiment_name'}] is not a PatternModel"

        if isinstance(model, PatternModelReturnsSLM):
            slm_pattern = model.generate(data_dock)

            self.slm.put(CameraPattern(experiment_name, slm_pattern, slm_coords=True))

        else:

            pattern = model.generate(data_dock)

            self.slm.put(CameraPattern(experiment_name, pattern, slm_coords=False, binning=model.binning))

    def check(self, experiment_name):

        dock: DataDock = self.docks.get(experiment_name)

        # print(dock.get_awaiting())

        if dock.check_complete():
            self.run_model(experiment_name)

    def handle_message(self, message: Message):

        # print(message.message)

        match message.message:

            case "request_pattern":

                assert isinstance(message, RequestPattern)

                req = message.requirements
                name = message.experiment_name
                t = message.t

                dock = DataDock(t, req)

                self.docks[name] = dock

                self.check(name)

            # case "request_method":
            #
            #     assert isinstance(message, RequestMethod)
            #     requirements = self.request_model(message.experiment)
            #
            #     returned_message = PatternRequirements(requirements)
            #
            #     self.manager.put(returned_message)
            #
            #     return None

            case _:
                raise NotImplementedError

    def process(self):
        while True:
            if not self.inbox.empty():
                msg = self.inbox.get()

                self.handle_message(msg)

            if not self.from_raw.empty():
                data = self.from_raw.get()

                assert isinstance(data, AcquisitionData)
                name = data.event.experiment_name

                self.docks[name].add_raw(data)

                self.check(name)

            if not self.from_seg.empty():
                data = self.from_seg.get()

                assert isinstance(data, SegmentationData)
                name = data.event.experiment_name

                self.docks[name].add_seg(data)

                self.check(name)


if __name__ == "__main__":
    print(type(PatternModel))
    air = AcquiredImageRequest(uuid4(), True, False)
    print(air)

# todo: what is the dynamic range of patterns? [0-1] float16?
# todo: dummy model
# todo: register model
# todo: intialize pattern process with pattern shape
# todo: handle raw and segmentation input
