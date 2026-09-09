import logging
from threading import Event

from .base_process import PipelineProcess
from .datatypes import AcquisitionData, CameraPattern
from .experiments import Experiment
from .messages import (
    Message,
    PatternParamsResultMessage,
    SettingsRequestMessage,
    StreamCloseMessage,
)
from .patterns import (
    AcquiredImageRequest,
    CameraProperties,
    DataDock,
    PatternContext,
    PatternMethod,
    PatternMethodReturnsSLM,
    known_models,
)
from .patterns.pattern import ExperimentState
from .plan import requirement_kinds
from .queues import AllQueues
from .router import Subscription

logger = logging.getLogger(__name__)


class PatternProcess(PipelineProcess):
    """
    Generates patterns: the Manager announces each pattern-due timepoint with
    a :class:`RequestPattern`, the Router delivers the required raw / seg /
    tracks data at that cadence, and once the dock for that timepoint is
    complete the experiment's method runs and the result goes to the SLM
    buffer.
    """

    always_active = True

    def __init__(self, aq: AllQueues, stop_event: Event | None = None):
        super().__init__(stop_event, name="pattern")

        # per-instance copy so registrations do not leak between controllers
        self.known_models: dict[str, type[PatternMethod]] = dict(known_models)

        self.inbox = aq.manager_to_pattern
        self.slm = aq.pattern_to_slm
        self.to_manager = aq.pattern_to_manager
        # experiment name -> MicroscopePosition, for context.position() (set by the Controller)
        self.positions: dict = {}

        self.camera_properties = None
        self.initialized = False

        self.models = {}
        self.docks = {}
        self.experiments = {}
        # per-experiment memory: histories of deliveries and generated patterns
        self.states: dict[str, ExperimentState] = {}

        self.register_queue(self.inbox, self.handle_message_wrapper)

    def initialize(self, camera_properties: CameraProperties):
        self.camera_properties = camera_properties

        self.initialized = True

    # ------------------------------------------------------------ routing
    def subscriptions(self, plan) -> list[Subscription]:
        subs = []
        for name in plan.experiments:
            for channel, needs in plan.pattern_requirements(name).items():
                for kind in requirement_kinds(needs):
                    subs.append(Subscription(self.name, name, channel, kind, "pattern"))
        return subs

    # ------------------------------------------------------------ methods
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
        logger.debug(f"{experiment_name}: {method_name} kwargs {method_kwargs}")

        model = model_class(**method_kwargs)

        self.models[experiment_name] = model
        self.experiments[experiment_name] = experiment

        logger.info(f'initializing pattern model "{method_name}"')

        requirements = model.initialize(experiment)
        self.states[experiment_name] = ExperimentState(
            requirements, pattern_history=getattr(model, "pattern_history", 2)
        )
        return requirements

    def initialize_models(self):
        for experiment_name in self.models:
            model: PatternMethod = self.models[experiment_name]
            experiment: Experiment = self.experiments[experiment_name]
            model.configure_system(experiment_name, self.camera_properties, experiment)

    def register_method(self, model: type, name: str | None = None):
        assert issubclass(model, PatternMethod), (
            "model must be a subclass of PatternMethod"
        )

        model_name = model.name
        if name is not None:
            model_name = name

        if model_name in self.known_models:
            logger.warning(f"overwriting known model {model_name}")

        self.known_models[model_name] = model

    def run_model(self, experiment_name, dockname):
        data_dock = self.docks.pop(dockname)

        model = self.models.get(experiment_name, None)

        assert isinstance(model, PatternMethod), (
            f"self.models[{'experiment_name'}] is not a PatternMethod"
        )

        if model._experiment_ref is None:
            raise RuntimeError(
                f"Model {model.name} for {experiment_name} was not properly configured with an experiment reference."
            )

        state = self.states.get(experiment_name)
        if state is None:
            state = self.states[experiment_name] = ExperimentState(
                data_dock.requirements
            )
        state.absorb(data_dock, dockname[1])
        context = PatternContext(
            state, model._experiment_ref, position=self.positions.get(experiment_name)
        )

        if isinstance(model, PatternMethodReturnsSLM):
            pattern = model.generate(context)
            out = CameraPattern(experiment_name, pattern, slm_coords=True)
        else:
            pattern = model.generate(context)
            out = CameraPattern(
                experiment_name, pattern, slm_coords=False, binning=model.binning
            )
        state.record_pattern(out.pattern_id, pattern)
        self.slm.put(out)

        # setting changes the method asked for go to the Manager, which applies
        # them at the next timepoint boundary and records them
        if context.requests:
            self.to_manager.put(
                SettingsRequestMessage(experiment_name, dockname[1], context.requests)
            )

    def dock_key(self, experiment_name, t) -> tuple[str, int]:
        return (experiment_name, int(t))

    def check(self, experiment_name, dockname):
        dock: DataDock = self.docks.get(dockname)

        if dock.check_complete():
            self.run_model(experiment_name, dockname)

    # ----------------------------------------------------------- messages
    def handle_message(self, message: Message):
        match message.message:
            case "close":
                return False

            case "update_pattern_params":
                name = message.experiment_name
                model = self.models.get(name)
                if model is None:
                    result = PatternParamsResultMessage(
                        name, {}, dict.fromkeys(message.parameters, "no pattern method")
                    )
                else:
                    try:
                        applied, refused = model.update(**message.parameters)
                    except Exception as e:  # a method's own update() may raise
                        applied, refused = (
                            {},
                            dict.fromkeys(message.parameters, repr(e)),
                        )
                    result = PatternParamsResultMessage(name, applied, refused)
                self.to_manager.put(result)
                return False

            case "request_pattern":
                assert isinstance(message, RequestPattern)

                req = message.requirements
                name = message.experiment_name
                t_sec = message.time_sec
                t_index = message.t_index

                dock = DataDock(t_sec, req)
                dockname = self.dock_key(name, t_index)

                logger.debug(f"pattern request {dockname}")

                self.docks[dockname] = dock

                self.check(name, dockname)

                return False

            case _:
                raise NotImplementedError

    def handle_message_wrapper(self, message):
        if self.handle_message(message):
            return True
        return False

    def on_stream_end(self) -> bool:
        logger.info("pattern process: stream ended, closing the SLM buffer")
        self.slm.put(StreamCloseMessage())
        return super().on_stream_end()

    # --------------------------------------------------------------- data
    def handle_data(self, data):
        assert isinstance(data, AcquisitionData), (
            f"pattern process received {type(data)}"
        )
        name = data.event.experiment_name
        t_index = data.event.t_index

        dockname = self.dock_key(name, t_index)

        dock = self.docks.get(dockname)
        if dock is None:
            logger.warning(
                f"received {data.kind} data for {dockname} but no pattern was "
                "requested for that timepoint; dropping it"
            )
            return

        dock.add(data)

        self.check(name, dockname)


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
        # absolute timepoint, matching AcquisitionEvent.t_index of the data expected
        self.t_index = t_index
        self.time_sec = time_sec
        self.experiment_name = experiment_name
        self.requirements = requirements
