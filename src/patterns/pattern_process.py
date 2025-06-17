import logging

from .. import AllQueues, CameraProperties
from ..datatypes import CameraPattern, AcquisitionData, SegmentationData
from ..experiments import Experiment
from ..messages import Message
from .pattern import PatternReview, PatternModel, PatternModelReturnsSLM, DataDock, AcquiredImageRequest
from .bar_patterns import BouncingBarPattern, BarPatternBase
from .static_patterns import CirclePattern


class PatternProcess:

    known_models = {
        "circle": CirclePattern,
        "bar": BarPatternBase,
        "pattern_review": PatternReview,
        "bar_bounce": BouncingBarPattern,

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

        if dock.check_complete():
            self.run_model(experiment_name)

    def handle_message(self, message: Message):

        match message.message:

            case "request_pattern":

                assert isinstance(message, RequestPattern)

                req = message.requirements
                name = message.experiment_name
                t = message.t

                dock = DataDock(t, req)

                self.docks[name] = dock

                self.check(name)

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


