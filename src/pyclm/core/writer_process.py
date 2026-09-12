"""
The writer: a pipeline consumer that hands every frame, label image and
track table it receives to the configured :class:`~pyclm.core.storage.FrameWriter`.

It subscribes to every raw frame (so stimulation events and DMD patterns are
recorded even when the camera frame is not saved) and, with ``demand=False``,
to the segmentation and tracking output of channels whose method has
``save = true``: those subscriptions record what is produced anyway and never
cause a stage to run (docs/stage3-router-design.md §3.4).
"""

import logging
from pathlib import Path
from threading import Event

from .base_process import PipelineProcess
from .core_interface import MicroscopeCoreInterface
from .kinds import base_kind, seg_kind
from .plan import AcquisitionPlan
from .router import Subscription
from .storage import FrameWriter, HDF5WriterV1

logger = logging.getLogger(__name__)


class WriterProcess(PipelineProcess):
    always_active = True

    def __init__(
        self,
        base_path: Path | None = None,
        stop_event: Event | None = None,
        writer: FrameWriter | None = None,
    ):
        super().__init__(stop_event, name="writer")
        self.base_path = Path(base_path) if base_path is not None else Path.cwd()
        self.writer: FrameWriter = writer if writer is not None else HDF5WriterV1()
        self.plan: AcquisitionPlan | None = None

    @property
    def open_files(self) -> dict:
        """Open HDF5 handles when the HDF5 writer is in use (empty otherwise)."""
        return getattr(self.writer, "open_files", {})

    @property
    def dropped_frames(self) -> int:
        return getattr(self.writer, "dropped_frames", 0)

    # ------------------------------------------------------------ routing
    def subscriptions(self, plan: AcquisitionPlan) -> list[Subscription]:
        subs = []
        for name in plan.experiments:
            experiment = plan.schedule.experiments[name]
            record = {
                seg_kind(seg): bool(cfg.save)
                for seg, cfg in experiment.segmentations.items()
            }
            tracking = getattr(experiment, "tracking", None)
            record["tracks"] = bool(tracking is not None and tracking.save)
            stim = plan.stim_channel(name)
            for channel in plan.channels(name):
                subs.append(Subscription(self.name, name, channel, "raw", demand=False))
                if channel == stim:
                    subs.append(
                        Subscription(self.name, name, channel, "skipped", demand=False)
                    )
                for kind, wanted in record.items():
                    if wanted:
                        subs.append(
                            Subscription(self.name, name, channel, kind, demand=False)
                        )
        return subs

    # ---------------------------------------------------------- lifecycle
    def initialize(
        self,
        plan: AcquisitionPlan,
        core: MicroscopeCoreInterface,
        affine_transform=None,
        slm_shape=None,
    ):
        """Create the outputs for the plan. Returns the (path, layer) pairs for the GUI."""
        self.plan = plan
        recorded = routing = None
        if self.router is not None:
            recorded = self.router.deliveries_to(self.name)
            routing = self.router.as_dict()
        return self.writer.open(
            plan,
            core,
            self.base_path,
            affine_transform,
            slm_shape,
            recorded=recorded,
            routing=routing,
        )

    def close_files(self):
        """Finalise and close all outputs (idempotent)."""
        self.writer.close()

    def process(self):
        """Run the poll loop and close the outputs however the loop exits."""
        try:
            super().process()
        finally:
            self.close_files()

    def on_stream_end(self) -> bool:
        self.close_files()
        return super().on_stream_end()

    # ------------------------------------------------------------ writing
    def write_data(self, data):
        kind = base_kind(getattr(data, "kind", "raw"))
        if kind == "skipped":
            self.writer.write_skipped(data)
        elif kind == "seg":
            self.writer.write_labels(data)
        elif kind == "tracks":
            self.writer.write_tracks(data)
        else:
            self.writer.write_frame(data)

    def handle_data(self, data):
        self.write_data(data)


# the writer was the "microscope outbox" before Stage 3
MicroscopeOutbox = WriterProcess
