import logging

from .. import AllQueues
from ..datatypes import AcquisitionData, SegmentationData
from ..experiments import Experiment
from ..messages import Message
from ..segmentation import SegmentationModel


class SegmentationProcess:

    known_models = {
    }

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_seg
        self.manager = aq.seg_to_manager

        self.from_raw = aq.outbox_to_seg
        self.to_outbox = aq.seg_to_outbox

        self.to_pattern = aq.seg_to_pattern

        self.initialized = False

        self.models = {}

        self.accommodated_requests = []
        self.shared_resources = dict()

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

        this_resource_request = model.request_resource()

        if this_resource_request:
            self.handle_resource_request(model, this_resource_request)

        self.models[experiment_name] = model

    def handle_resource_request(self, model, request):
        preexisting_resource = None

        for accommodated_resource_request in self.accommodated_requests:
            if accommodated_resource_request == request:
                preexisting_resource = accommodated_resource_request.request_id

        if preexisting_resource:
            model.provide_resource(self.shared_resources[preexisting_resource])

        else:
            # initialize the resource
            resource_class = request.resource
            kwargs = request.init_kwargs

            resource = resource_class(**kwargs)

            # keep track of the requested resource for future requests
            self.accommodated_requests.append(request)

            this_request_id = request.request_id
            self.shared_resources[this_request_id] = resource

            # provide the resource to the model
            model.provide_resource(resource)
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

        event = aq_data.event
        name = event.experiment_name

        # pass data to pattern process for pattern gen
        seg_data = self.run_model(name, aq_data)
        self.to_pattern.put(seg_data)

        # pass data to outbox for saving
        if event.save_seg:
            self.to_outbox.put(seg_data)


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
