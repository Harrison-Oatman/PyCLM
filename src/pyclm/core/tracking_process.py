"""
The tracking process: produces ``tracks`` from ``seg`` for every channel
some consumer demands tracks of. It is ``continuous``: once it runs for a
channel it receives every acquired frame's segmentation, because a tracker
cannot link objects across frames it never saw.

The first consumer added through the Router as a registration rather than
a surgery (docs/stage3-router-design.md §5).
"""

import logging
from threading import Event
from typing import ClassVar

import numpy as np

from .base_process import PipelineProcess
from .datatypes import SegmentationData, TrackingData
from .experiments import Experiment
from .kinds import seg_kind
from .tracking import TrackingMethod, known_tracking_methods

logger = logging.getLogger(__name__)


class TrackingProcess(PipelineProcess):
    produces: ClassVar[dict[str, tuple[str, ...]]] = {"tracks": ("seg",)}
    continuous = True

    def __init__(self, stop_event: Event | None = None):
        super().__init__(stop_event, name="tracking")
        # per-instance copy so registrations do not leak between controllers
        self.known_methods: dict[str, type[TrackingMethod]] = dict(
            known_tracking_methods
        )
        # (experiment name, channel id) -> method instance
        self.methods: dict[tuple, TrackingMethod] = {}
        # the stage x / y of each method's last frame, to follow stage moves
        self._stage: dict[tuple, tuple[float, float]] = {}
        # unbinned camera pixels -> stage um (MicroManager's pixel-size
        # affine), set by the Controller; None: pixel size x identity
        self.camera_to_stage = None

    def can_produce(self, kind: str, experiment: Experiment, channel: str) -> bool:
        tracking = getattr(experiment, "tracking", None)
        return (
            kind == "tracks" and tracking is not None and tracking.method_name != "none"
        )

    def inputs(
        self, kind: str, experiment: Experiment, channel: str
    ) -> tuple[str, ...]:
        """Tracking links the segmentation named by ``[tracking] segmentation`` (default: the default table)."""
        name = getattr(experiment.tracking, "segmentation", None)
        return (seg_kind(name),)

    def register_method(self, method: type, name: str | None = None):
        assert issubclass(method, TrackingMethod), (
            "method must be a subclass of TrackingMethod"
        )
        method_name = method.name if name is None else name
        if method_name in self.known_methods:
            logger.warning(f"overwriting known tracking method {method_name}")
        self.known_methods[method_name] = method

    def request_method(self, experiment: Experiment, channel: str) -> TrackingMethod:
        """Construct the experiment's tracking method for one channel."""
        method_name = experiment.tracking.method_name
        method_class = self.known_methods.get(method_name)
        assert method_class is not None, (
            f"tracking method {method_name} is not registered"
        )
        assert issubclass(method_class, TrackingMethod), (
            f"{method_name} is not a TrackingMethod"
        )

        name = experiment.experiment_name
        cfg = (
            experiment.stimulation
            if channel not in experiment.channels
            else experiment.channels[channel]
        )
        method = method_class(name, channel, **experiment.tracking.kwargs)
        self.methods[(name, cfg.channel_id)] = method
        logger.info(f'{name}/{channel}: initialized tracking method "{method_name}"')
        return method

    def handle_data(self, data):
        assert isinstance(data, SegmentationData), (
            f"tracking received {type(data)}, expected SegmentationData"
        )
        event = data.event
        method = self.methods.get((event.experiment_name, event.channel_id))
        if method is None:
            logger.warning(
                f"no tracking method for {event.experiment_name}/{event.index.get('c')}; "
                "dropping segmentation"
            )
            return

        pixel_size_um = event.pixel_width_um
        if not pixel_size_um:
            pixel_size_um = 1.0
        self._follow_stage(method, event, float(pixel_size_um))
        labels, rows = method.track(
            np.asarray(data.data), event.t_index, float(pixel_size_um)
        )
        logger.debug(
            f"{event.experiment_name}/{event.index.get('c')} t={event.t_index}: "
            f"{len(rows)} tracked objects"
        )
        self.publish(TrackingData(event, labels, rows))

    def stage_to_pixels(
        self, dx_um: float, dy_um: float, pixel_size_um: float, binning: int = 1
    ) -> tuple[float, float]:
        """A stage displacement as ``(dy, dx)`` in the frame's pixels."""
        if self.camera_to_stage is None:
            return dy_um / pixel_size_um, dx_um / pixel_size_um
        m = np.asarray(self.camera_to_stage, dtype=float) * max(1, int(binning))
        dx_px, dy_px = np.linalg.solve(m, np.array([dx_um, dy_um]))
        return float(dy_px), float(dx_px)

    def _follow_stage(self, method: TrackingMethod, event, pixel_size_um: float):
        """
        Tell the method how far the objects moved in the image because the
        stage moved since its last frame (a pattern method's set_position):
        a stage move of ``d`` carries the content by ``-d``.
        """
        xy = getattr(event, "stage_xy", None)
        key = (event.experiment_name, event.channel_id)
        previous = self._stage.get(key)
        if xy is None:
            return
        self._stage[key] = xy
        if previous is None or previous == xy:
            return
        dy, dx = self.stage_to_pixels(
            xy[0] - previous[0],
            xy[1] - previous[1],
            pixel_size_um,
            getattr(event, "binning", 1),
        )
        logger.info(
            f"{event.experiment_name}: stage moved by "
            f"({xy[0] - previous[0]:.2f}, {xy[1] - previous[1]:.2f}) um; "
            f"shifting the tracker by ({-dy:.1f}, {-dx:.1f}) px"
        )
        method.shift(-dy, -dx)
