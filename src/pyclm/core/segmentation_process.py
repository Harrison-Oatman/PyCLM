import logging
from threading import Event
from typing import ClassVar

from .base_process import PipelineProcess
from .datatypes import AcquisitionData, SegmentationData
from .experiments import Experiment
from .kinds import DEFAULT_SEGMENTATION, base_kind, seg_name
from .segmentation import SegmentationMethod
from .segmentation.cellpose_segmentation import (
    CellposeSegmentationMethod,
    EmbryoSegmentationMethod,
)

logger = logging.getLogger(__name__)


class SegmentationProcess(PipelineProcess):
    """
    Produces ``seg`` from ``raw``: runs the experiment's segmentation methods
    on every frame the Router delivers and publishes one label image per
    demanded ``[segmentation]`` table (the default one travels as ``seg``, a
    named ``[segmentation.<name>]`` as ``seg:<name>``).

    Runs only where some consumer demands a segmentation (the Router
    decides; the Controller then calls :meth:`request_method` per name).
    """

    produces: ClassVar[dict[str, tuple[str, ...]]] = {"seg": ("raw",)}

    default_models: ClassVar[dict[str, type[SegmentationMethod]]] = {
        "cellpose": CellposeSegmentationMethod,
        "embryo_resizing": EmbryoSegmentationMethod,
    }

    def __init__(self, stop_event: Event | None = None):
        super().__init__(stop_event, name="segmentation")

        # per-instance copy so registrations do not leak between controllers
        self.known_models: dict[str, type[SegmentationMethod]] = dict(
            self.default_models
        )

        self.initialized = False

        self.models = {}

        self.accommodated_requests = []
        self.shared_resources = dict()

    def initialize(self):
        self.initialized = True

    def can_produce(self, kind: str, experiment: Experiment, channel: str) -> bool:
        if base_kind(kind) != "seg":
            return False
        cfg = experiment.segmentations.get(seg_name(kind))
        return cfg is not None and cfg.method_name != "none"

    def register_method(self, method: type, name: str | None = None):
        assert issubclass(method, SegmentationMethod), (
            "model must be a subclass of PatternModel"
        )

        model_name = method.name
        if name is not None:
            model_name = name

        if model_name in self.known_models:
            logger.warning(f"overwriting known model {model_name}")

        self.known_models[model_name] = method

    def request_method(self, experiment: Experiment, name: str = DEFAULT_SEGMENTATION):
        """Construct the method of one ``[segmentation]`` table for an experiment."""
        cfg = experiment.segmentations[name]
        method_name = cfg.method_name

        model_class: type = self.known_models.get(method_name)

        assert model_class is not None, (
            f"method {method_name} is not a registered model"
        )
        assert issubclass(model_class, SegmentationMethod), (
            f"{method_name} is not a SegmentationModel"
        )

        experiment_name = experiment.experiment_name
        method_kwargs = cfg.kwargs

        model = model_class(experiment_name, **method_kwargs)

        this_resource_request = model.request_resource()

        if this_resource_request:
            self.handle_resource_request(model, this_resource_request)

        self.models[(experiment_name, name)] = model

    def handle_resource_request(self, model, request):
        preexisting_resource = None

        for accommodated_resource_request in self.accommodated_requests:
            if accommodated_resource_request == request:
                preexisting_resource = accommodated_resource_request.request_id

        if preexisting_resource:
            logger.info("using existing shared segmentation resource")
            model.provide_resource(self.shared_resources[preexisting_resource])

        else:
            logger.info("creating new shared segmentation resource")
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

    def run_model(
        self,
        experiment_name,
        aq_data: AcquisitionData,
        name: str = DEFAULT_SEGMENTATION,
    ) -> SegmentationData:
        model = self.models.get((experiment_name, name), None)

        assert isinstance(model, SegmentationMethod), (
            f"self.models[{(experiment_name, name)}] is not a SegmentationModel"
        )

        data_to_seg = aq_data.data
        segmented = model.segment(data_to_seg)

        event = aq_data.event
        seg_data = SegmentationData(event, segmented, name)

        return seg_data

    def segmentations_for(
        self, experiment_name: str, channel: str
    ) -> list[tuple[str, str]]:
        """
        ``[(segmentation name, cadence)]`` to run on a frame of that channel:
        from the routing table when attached to a router, else every method
        constructed for the experiment.
        """
        produced = getattr(self.router, "produced_by", None)
        if produced is not None:
            wanted = produced(self.name).get((experiment_name, channel))
            if wanted is not None:
                return [(seg_name(kind), cadence) for kind, cadence in wanted]
        return [(n, "always") for (e, n) in self.models if e == experiment_name]

    def handle_data(self, data):
        assert isinstance(data, AcquisitionData), (
            f"segmentation received {type(data)}, expected AcquisitionData"
        )
        event = data.event
        experiment_name = event.experiment_name
        channel = event.index.get("c")

        for name, cadence in self.segmentations_for(experiment_name, channel):
            if cadence == "pattern" and not self.router.plan.pattern_due(
                experiment_name, event.t_index
            ):
                continue
            logger.debug(
                f"segmenting {experiment_name}/{channel} ({name}): t = {event.t_index}"
            )
            self.publish(self.run_model(experiment_name, data, name))
