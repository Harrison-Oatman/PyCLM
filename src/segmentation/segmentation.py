import logging
import numpy as np

from ..queues import AllQueues
from ..experiments import Experiment
from ..datatypes import AcquisitionData, SegmentationData
from ..messages import Message


class SegmentationModel:
    name = "base_model"

    def __init__(self, experiment_name, **kwargs):
        pass

    def segment(self, data: np.ndarray) -> np.ndarray:

        raise NotImplementedError("base_model does not implement segment")


class SegmentationProcess:

    known_models = {
    }

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_seg
        self.manager = aq.seg_to_manager

        self.from_raw = aq.outbox_to_seg
        self.to_pattern = aq.seg_to_pattern

        self.initialized = False

        self.models = {}

    def initialize(self):
        self.initialized = True

    def request_model(self, experiment: Experiment):

        method_name = experiment.pattern.method_name

        model_class: type = self.known_models.get(method_name)

        assert model_class is not None, f"method {method_name} is not a registered model"
        assert issubclass(model_class, SegmentationModel), f"{method_name} is not a PatternModel"

        experiment_name = experiment.experiment_name
        method_kwargs = experiment.pattern.kwargs

        model = model_class(experiment_name, **method_kwargs)

        self.models[experiment_name] = model

    def register_model(self, model: type):

        assert issubclass(model, SegmentationModel), "model must be a subclass of PatternModel"

        model_name = model.name

        if model_name in self.known_models:
            logging.warning(f"overwriting known model {model_name}")

        self.known_models[model_name] = model

    def run_model(self, experiment_name, aq_data: AcquisitionData) -> SegmentationData:

        model = self.models.get(experiment_name, None)

        assert isinstance(model, SegmentationModel), f"self.models[{'experiment_name'}] is not a PatternModel"

        data_to_seg = aq_data.data
        segmented = model.segment(data_to_seg)

        event = aq_data.event
        seg_data = SegmentationData(event, segmented)

        return seg_data

    def handle_segment_data(self, aq_data: AcquisitionData):

        name = aq_data.event.experiment_name

        seg_data = self.run_model(name, aq_data)

    def handle_message(self, message: Message):

        match message.message:

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
                self.handle_segment_data(data)
