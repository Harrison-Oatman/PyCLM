from ..queues import AllQueues
from ..experiments import Experiment, ImagingConfig
from ..messages import Message
from ..datatypes import AcquisitionData, SegmentationData, CameraPattern
from logging import warning
from typing import NamedTuple, Union
from uuid import UUID, uuid4
from collections import defaultdict
from pathlib import Path
from h5py import File
import numpy as np
from natsort import natsorted


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

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        raise NotImplementedError

    def generate(self, data_dock: DataDock) -> np.ndarray:
        raise NotImplementedError

class PatternModelReturnsSLM(PatternModel):

    pass


# class OpenLoopPatternModel(PatternModel):
#
#     name = "open loop"
#
#     def __init__(self, experiment_name, camera_properties, **kwargs):
#         super().__init__(experiment_name, camera_properties)
#
#
#     def initialize(self, experiment):
#         return []

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
        return []

    def generate(self, data_dock: DataDock):

        h, w = self.pattern_shape

        y_range = np.arange(h)
        x_range = np.arange(w)

        center_x = w / 2.
        center_y = h / 2.

        xx, yy = np.meshgrid(x_range, y_range)

        pattern = np.ones((h, w), dtype=np.float16)

        return (pattern * ((xx - center_x)**2 + (yy - center_y)**2 < self.rad**2)).astype(np.float16)


class BarPattern(PatternModel):
    """
    moves a bar along the y-axis
    """

    name = "bar"

    def __init__(self, experiment_name, camera_properties, duty_cycle=0.2, bar_speed=0.01, bar_width=40, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/sec
        :param bar_width: width in um
        """

        super().__init__(experiment_name, camera_properties)

        self.duty_cycle = duty_cycle
        self.bar_speed = bar_speed
        self.bar_width = bar_width
        self.L = bar_width / duty_cycle

    def initialize(self, experiment):
        return []

    def generate(self, data_dock: DataDock):

        t = data_dock.t

        px_um = self.pixel_size_um

        h, w = self.pattern_shape

        y_range = np.arange(h)
        x_range = np.arange(w)

        xx, yy = np.meshgrid(x_range, y_range)

        y_um = px_um*yy

        is_on = (((t*self.bar_speed - y_um) / self.L) % 1) < self.duty_cycle

        return is_on.astype(np.float16)


# class PatternDataDock:
#
#     def
#

class PatternProcess:

    known_models = {
        "circle": CirclePattern,
        "bar": BarPattern,
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
            warning(f"overwriting known model {model_name}")

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

            self.slm.put(CameraPattern(experiment_name, pattern))

    def check(self, experiment_name):

        dock: DataDock = self.docks.get(experiment_name)

        print(dock.get_awaiting())

        if dock.check_complete():
            self.run_model(experiment_name)

    def handle_message(self, message: Message):

        print(message.message)

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
