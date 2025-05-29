from ..queues import AllQueues


class SegmentationModel:

    def __init__(self):
        pass

    def segment(self):
        pass


class SegmentationProcess:

    known_models = {
    }

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_seg
        self.manager = aq.seg_to_manager

        self.from_raw = aq.outbox_to_seg
        self.to_seg = aq.seg_to_pattern

        self.initialized = False

        self.models = {}
        self.docks = {}

    def initialize(self):
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
