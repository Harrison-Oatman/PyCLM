import logging
from threading import Event
from typing import ClassVar

from .datatypes import AcquisitionData, CameraPattern, SegmentationData
from .experiments import Experiment
from .messages import Message, StreamCloseMessage
from .queues import AllQueues

logger = logging.getLogger(__name__)

from .base_process import BaseProcess
from .patterns import (
    AcquiredImageRequest,
    CameraProperties,
    DataDock,
    PatternContext,
    PatternMethod,
    PatternMethodReturnsSLM,
    known_models,
)


class PatternProcess(BaseProcess):
    known_models: ClassVar = known_models

    def __init__(self, aq: AllQueues, stop_event: Event | None = None):
        super().__init__(stop_event, name="pattern")

        self.inbox = aq.manager_to_pattern
        self.manager = aq.pattern_to_manager
        self.slm = aq.pattern_to_slm

        self.from_seg = aq.seg_to_pattern
        self.from_raw = aq.outbox_to_pattern

        self.stream_count = 0

        self.camera_properties = None
        self.initialized = False

        self.models = {}
        self.docks = {}

        self.register_queue(self.inbox, self.handle_message_wrapper)
        self.register_queue(self.from_raw, self.handle_from_raw)
        self.register_queue(self.from_seg, self.handle_from_seg)

    def initialize(self, camera_properties: CameraProperties):
        self.camera_properties = camera_properties

        self.initialized = True

    def request_method(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        method_name = experiment.pattern.method_name

        model_class: type = self.known_models.get(method_name)

        assert model_class is not None, (
            f"method {method_name} is not a registered method"
        )
        assert issubclass(model_class, PatternMethod), (
            f"{method_name} is not a PatternMethod"
        )

        experiment_name = experiment.experiment_name
        method_kwargs = experiment.pattern.kwargs

        model = model_class(experiment_name, self.camera_properties, **method_kwargs)

        self.models[experiment_name] = model

        logger.info(f'initializing pattern model "{method_name}"')

        return model.initialize(experiment)

    def register_method(self, model: type, name: str | None = None):
        assert issubclass(model, PatternMethod), (
            "model must be a subclass of PatternMethod"
        )

        model_name = model.name
        if name is not None:
            model_name = name

        if model_name in self.known_models:
            logging.warning(f"overwriting known model {model_name}")

        self.known_models[model_name] = model

    def run_model(self, experiment_name, dockname):
        data_dock = self.docks.pop(dockname)

        model = self.models.get(experiment_name, None)

        assert isinstance(model, PatternMethod), (
            f"self.models[{'experiment_name'}] is not a PatternMethod"
        )

        # Create context wrapper

        # We assume _experiment_ref is available. If not, we might crash, which is acceptable for alpha breakage.
        # Ideally, we ensure configure_system is called.
        if model._experiment_ref is None:
            raise RuntimeError(
                f"Model {model.name} for {experiment_name} was not properly configured with an experiment reference."
            )

        context = PatternContext(data_dock, model._experiment_ref)

        if isinstance(model, PatternMethodReturnsSLM):
            slm_pattern = model.generate(context)
            self.slm.put(CameraPattern(experiment_name, slm_pattern, slm_coords=True))

        else:
            pattern = model.generate(context)
            self.slm.put(
                CameraPattern(
                    experiment_name, pattern, slm_coords=False, binning=model.binning
                )
            )

    def dock_string(self, experiment_name, t):
        return f"{experiment_name}_{t:05d}"

    def check(self, experiment_name, dockname):
        dock: DataDock = self.docks.get(dockname)

        if dock.check_complete():
            self.run_model(experiment_name, dockname)

    def handle_message(self, message: Message):
        match message.message:
            case "close":
                return False

            case "stream_close":
                print("pattern received stream close")

                self.stream_count += 1
                if self.stream_count >= 2:
                    out_msg = StreamCloseMessage()
                    self.slm.put(out_msg)
                    return True
                return False

            case "request_pattern":
                assert isinstance(message, RequestPattern)

                req = message.requirements
                name = message.experiment_name
                t_sec = message.time_sec
                t_index = message.t_index

                dock = DataDock(t_sec, req)
                dockname = self.dock_string(name, t_index)

                print(f"pattern request {dockname}")

                self.docks[dockname] = dock

                self.check(name, dockname)

                return False

            case _:
                raise NotImplementedError

    def handle_message_wrapper(self, message):
        if self.handle_message(message):
            return True
        return False

    def handle_from_raw(self, data):
        if isinstance(data, Message):
            if self.handle_message(data):
                return True
        else:
            assert isinstance(data, AcquisitionData)
            name = data.event.experiment_name
            t_index = data.event.t_index

            dockname = self.dock_string(name, t_index)

            self.docks[dockname].add_raw(data)

            self.check(name, dockname)
        return False

    def handle_from_seg(self, data):
        if isinstance(data, Message):
            if self.handle_message(data):
                return True
        else:
            assert isinstance(data, SegmentationData)
            name = data.event.experiment_name
            t_index = data.event.t_index

            dockname = self.dock_string(name, t_index)

            print(f"seg found {dockname}")

            self.docks[dockname].add_seg(data)

            self.check(name, dockname)
        return False


class RequestPattern(Message):
    """
    Used by manager to inform pattern process of upcoming pattern generation
    Pattern process will use this information to collect and store incoming
    raw and segmented data
    """

    message = "request_pattern"

    def __init__(
        self,
        t_index,
        time_sec,
        experiment_name: str,
        requirements: list[AcquiredImageRequest],
    ):
        self.t_index = t_index
        self.time_sec = time_sec
        self.experiment_name = experiment_name
        self.requirements = requirements
