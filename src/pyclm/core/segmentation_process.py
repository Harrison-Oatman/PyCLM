import logging
from threading import Event
from typing import ClassVar

import numpy as np

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

        # multi-channel segmentations: the frames of one timepoint gathered so
        # far, {(experiment, segmented channel, name, t): {channel: frame}}
        self._pending: dict[tuple, dict[str, AcquisitionData]] = {}
        # (experiment, name) -> the [segmentation] table's channels, or None
        self._channels: dict[tuple[str, str], list[str] | None] = {}

    def initialize(self):
        self.initialized = True

    def can_produce(self, kind: str, experiment: Experiment, channel: str) -> bool:
        if base_kind(kind) != "seg":
            return False
        cfg = experiment.segmentations.get(seg_name(kind))
        if cfg is None or cfg.method_name == "none":
            return False
        # a multi-channel table segments only the channels it lists
        channels = getattr(cfg, "channels", None)
        return not channels or channel in channels

    def inputs(self, kind: str, experiment: Experiment, channel: str) -> list:
        """The raw frames a segmentation of ``channel`` reads: its own, or every listed channel's."""
        cfg = experiment.segmentations.get(seg_name(kind))
        channels = getattr(cfg, "channels", None)
        if not channels or channel not in channels:
            return ["raw"]
        return [("raw", c) for c in channels]

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
        self._channels[(experiment_name, name)] = getattr(cfg, "channels", None)

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
        frames: list[AcquisitionData] | None = None,
    ) -> SegmentationData:
        """
        Segment ``aq_data`` (whose event the label image carries), or, for a
        multi-channel table, ``frames`` (one per listed channel, in order)
        stacked into one ``(height, width, channel)`` image.
        """
        model = self.models.get((experiment_name, name), None)

        assert isinstance(model, SegmentationMethod), (
            f"self.models[{(experiment_name, name)}] is not a SegmentationModel"
        )

        if frames is None or len(frames) == 1:
            data_to_seg = aq_data.data if frames is None else frames[0].data
        else:
            data_to_seg = stack_channels(frames)
        segmented = model.segment(data_to_seg)

        event = aq_data.event
        seg_data = SegmentationData(event, segmented, name)

        return seg_data

    def segmentations_for(
        self, experiment_name: str, channel: str
    ) -> list[tuple[str, str, str]]:
        """
        ``[(segmented channel, segmentation name, cadence)]`` a frame of
        ``channel`` feeds: from the routing table when attached to a router,
        else every method constructed for the experiment. A multi-channel
        table is fed by each of its channels but segments the channel a
        consumer asked for.
        """
        produced = getattr(self.router, "produced_by", None)
        if produced is None:
            return [
                (channel, n, "always") for (e, n) in self.models if e == experiment_name
            ]
        out = []
        for (exp, segmented), wanted in produced(self.name).items():
            if exp != experiment_name:
                continue
            for kind, cadence in wanted:
                name = seg_name(kind)
                listed = self._channels.get((exp, name))
                inputs = listed if listed else [segmented]
                if channel in inputs:
                    out.append((segmented, name, cadence))
        return out

    def handle_data(self, data):
        assert isinstance(data, AcquisitionData), (
            f"segmentation received {type(data)}, expected AcquisitionData"
        )
        event = data.event
        experiment_name = event.experiment_name
        channel = event.index.get("c")

        for segmented, name, cadence in self.segmentations_for(
            experiment_name, channel
        ):
            if cadence == "pattern" and not self.router.plan.pattern_due(
                experiment_name, event.t_index
            ):
                continue
            listed = self._channels.get((experiment_name, name))
            if not listed or len(listed) == 1:
                logger.debug(
                    f"segmenting {experiment_name}/{channel} ({name}): t = {event.t_index}"
                )
                self.publish(self.run_model(experiment_name, data, name))
                continue
            frames = self._gather(experiment_name, segmented, name, listed, data)
            if frames is None:
                continue
            logger.debug(
                f"segmenting {experiment_name}/{segmented} ({name}) from "
                f"{listed}: t = {event.t_index}"
            )
            own = frames[listed.index(segmented)]
            self.publish(self.run_model(experiment_name, own, name, frames))

    def _gather(self, experiment_name, segmented, name, listed, data):
        """Hold ``data`` until every listed channel of its timepoint is in; then the frames, in order."""
        t = int(data.event.t_index)
        key = (experiment_name, segmented, name, t)
        have = self._pending.setdefault(key, {})
        have[data.event.index.get("c")] = data
        # an earlier timepoint still missing a channel will not complete now
        for old in [k for k in self._pending if k[:3] == key[:3] and k[3] < t]:
            missing = sorted(set(listed) - set(self._pending.pop(old)))
            logger.warning(
                f"{experiment_name} t = {old[3]}: segmentation '{name}' never got "
                f"channel(s) {missing}; that timepoint is not segmented"
            )
        if not all(c in have for c in listed):
            return None
        del self._pending[key]
        return [have[c] for c in listed]


def stack_channels(frames: list[AcquisitionData]) -> np.ndarray:
    """One ``(height, width, channel)`` image from frames of one timepoint, in order."""
    arrays = [np.asarray(f.data) for f in frames]
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        detail = ", ".join(
            f"{f.event.index.get('c')}: {a.shape}"
            for f, a in zip(frames, arrays, strict=True)
        )
        raise ValueError(
            f"a multi-channel segmentation needs frames of one shape ({detail}); "
            "give the channels the same binning"
        )
    return np.stack(arrays, axis=-1)
