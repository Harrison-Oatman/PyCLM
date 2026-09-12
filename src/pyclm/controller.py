import json
import logging
import math
import traceback
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from threading import Event

import numpy as np

from .core import (
    ROI,
    AllQueues,
    CameraProperties,
    ExperimentSchedule,
    Manager,
    MicroscopeProcess,
    PatternProcess,
    SegmentationProcess,
    SLMBuffer,
    WriterProcess,
)
from .core.base_process import PipelineProcess
from .core.grid import geometry_for
from .core.kinds import DEFAULT_SEGMENTATION, seg_name
from .core.plan import PLAN_FILENAME, AcquisitionPlan
from .core.position_mover import PositionMover
from .core.real_core import RealMicroscopeCore
from .core.router import Router
from .core.storage import make_writer
from .core.storage.events import EventLog
from .core.tracking_process import TrackingProcess
from .core.virtual_microscope.simulated_core import SimulatedMicroscopeCore

logger = logging.getLogger(__name__)


class Controller:
    """
    Owns the pipeline processes and runs them as threads.

    The fixed processes are the microscope (produces raw frames), the manager
    (timing), the SLM buffer, the writer, segmentation and pattern
    generation. Extra consumers or producers are added with
    :meth:`add_process` before :meth:`initialize`, which builds the
    :class:`~pyclm.core.router.Router` from every process's declared
    requirements and starts only the processes that will receive data.
    """

    def __init__(
        self,
        config="MMConfig_demo.cfg",
        dry=False,
        position_mover: PositionMover | None = None,
        dry_image_source: Path | None = None,
        settle_time_s: float = 1.0,
        storage_format: str = "ome-zarr",
        pattern_policy: str = "on_change",
    ):
        if not dry:
            # Applies if config specifies that a real microscope is in use
            self.core = RealMicroscopeCore()
        else:
            if dry_image_source is None:
                raise ValueError("dry_image_source must be provided when dry=True.")
            self.core = SimulatedMicroscopeCore(
                dry_image_source,
                pixel_size_um=getattr(dry_image_source, "pixel_size_um", 0.33),
                slm_device="SimulatedSLM",
            )
            self.dry_binning = int(getattr(dry_image_source, "binning", 1))
        self.core.loadSystemConfiguration(config)
        self.all_queues = AllQueues()

        self.stop_event = Event()

        self.microscope = MicroscopeProcess(
            core=self.core,
            aq=self.all_queues,
            position_mover=position_mover,
            stop_event=self.stop_event,
            settle_time_s=settle_time_s,
        )
        self.manager = Manager(aq=self.all_queues, stop_event=self.stop_event)
        self.writer_process = WriterProcess(
            stop_event=self.stop_event,
            writer=make_writer(storage_format, pattern_policy),
        )
        # the writer was the "outbox" before Stage 3; the old name still works
        self.outbox = self.writer_process
        self.slm_buffer = SLMBuffer(aq=self.all_queues, stop_event=self.stop_event)
        self.segmentation = SegmentationProcess(stop_event=self.stop_event)
        self.tracking = TrackingProcess(stop_event=self.stop_event)
        self.pattern = PatternProcess(aq=self.all_queues, stop_event=self.stop_event)

        # processes the router manages, in registration order (raw producer first)
        self.pipeline_processes: list = [
            self.microscope,
            self.writer_process,
            self.segmentation,
            self.tracking,
            self.pattern,
        ]
        self.router: Router | None = None

        # everything that could run; narrowed to the active set by initialize()
        self.processes = [
            self.microscope,
            self.manager,
            self.writer_process,
            self.slm_buffer,
            self.segmentation,
            self.tracking,
            self.pattern,
        ]

        self.camera_properties = None
        self.settle_time_s = settle_time_s

        self.plan: AcquisitionPlan | None = None
        self.all_layers = None
        self.t_gcd = 1

    def register_pattern_method(self, name: str, method: type):
        self.pattern.register_method(method, name)

    def register_segmentation_method(self, name: str, method: type):
        self.segmentation.register_method(method, name)

    def register_tracking_method(self, name: str, method: type):
        self.tracking.register_method(method, name)

    def add_process(self, process: PipelineProcess):
        """
        Register an extra pipeline process (a consumer, a producer, or both)
        before :meth:`initialize`. It declares what it wants and makes; the
        router wires it in and derives its shutdown from the same table.
        """
        if self.router is not None:
            raise RuntimeError("add_process must be called before initialize()")
        if not isinstance(process, PipelineProcess):
            raise TypeError(
                f"add_process expects a PipelineProcess, got {type(process).__name__}"
            )
        if process.stop_event is None:
            process.stop_event = self.stop_event
        self.pipeline_processes.append(process)

    def initialize(
        self,
        schedule: ExperimentSchedule,
        slm_shape: tuple[int, int],
        affine_transform: np.ndarray,
        out_path: Path,
        camera_roi=None,
    ):
        """
        :param camera_roi: ``[x, y, width, height]`` in unbinned pixels to set on
            the camera before anything is measured; None leaves the camera as it is
        """
        # refuse to run before any models are loaded if outputs already exist
        out_path = Path(out_path)
        planned = self.writer_process.writer.planned_paths(
            schedule.experiment_names, out_path
        )
        existing = [path for path in planned.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "output already exists; move or delete before re-running: "
                + ", ".join(str(path) for path in existing)
            )

        if isinstance(self.core, SimulatedMicroscopeCore):
            self.core._slm_h, self.core._slm_w = int(slm_shape[0]), int(slm_shape[1])
            binning = getattr(self, "dry_binning", 1)
            if binning != 1:
                # the TIFs are binned relative to the camera the affine was
                # calibrated for: scale the camera -> SLM affine to match
                affine_transform = np.array(affine_transform, dtype=np.float32)
                affine_transform[:, :2] *= binning
                logger.info(
                    f"dry run: affine transform scaled by the source binning {binning}"
                )

        self.microscope.set_binning(1)
        if camera_roi is not None:
            self.core.setROI(*[int(v) for v in camera_roi])
            logger.info(f"camera ROI set to {tuple(int(v) for v in camera_roi)}")

        camera_roi = ROI(*self.core.getROI())
        camera_resolution = self.core.getPixelSizeUm()

        self.camera_properties = CameraProperties(camera_roi, camera_resolution)

        logger.info(f"camera properties: {self.camera_properties}")

        # grid positions: the tiles' layout in camera pixels, shared by the
        # pattern process (stitched pattern shape), the SLM buffer (cutting)
        # and the writer (stitched frame shape) through the position object
        self.grids = {}
        for name, position in schedule.positions.items():
            geometry = geometry_for(position, camera_roi, camera_resolution)
            position.geometry = geometry
            if geometry is not None:
                self.grids[name] = geometry
                oy, ox = geometry.overlap(1)
                logger.info(
                    f"{name}: grid of {geometry.rows} x {geometry.columns} tiles, "
                    f"stitched frame {geometry.shape(1)} px, overlap {oy} x {ox} px"
                )
        self.pattern.grids = self.grids

        self.pattern.initialize(self.camera_properties)

        pattern_requirements = {}
        t_seen = set()
        for name, experiment in schedule.experiments.items():
            pattern_requirements[name] = self.pattern.request_method(experiment)

            for channel in experiment.channels.values():
                t_seen.add(channel.every_t)

        if len(t_seen) == 1:
            self.t_gcd = t_seen.pop()

        elif len(t_seen) > 1:
            self.t_gcd = math.gcd(*t_seen)

        self.pattern.initialize_models()

        # the plan is the single description of what is acquired when
        plan = AcquisitionPlan.from_schedule(schedule, pattern_requirements)
        self.plan = plan
        plan_path = plan.to_yaml(out_path / PLAN_FILENAME)
        logger.info(f"wrote acquisition plan to {plan_path}: {plan}")

        over = plan.over_budget(settle_s=self.settle_time_s)
        if over:
            worst_t, worst = max(over, key=lambda item: item[1])
            logger.warning(
                f"{len(over)} of {plan.timepoints} timepoints are estimated to take "
                f"longer than the {plan.interval_s:.1f}s interval (worst: t={worst_t}, "
                f"{worst:.1f}s with settle time {self.settle_time_s}s); acquisitions "
                "will run late"
            )

        # who receives what: resolved once from the methods' requirements
        router = Router(plan)
        for process in self.pipeline_processes:
            router.add(process)
        router.resolve()
        self.router = router
        self._instantiate_producers(schedule, router)
        logger.info(f"routing table: {json.dumps(router.as_dict())}")

        self.event_log = EventLog(out_path / "events.parquet")
        self.manager.initialize(
            plan,
            event_log=self.event_log,
            status_path=out_path / "status.json",
            health=self.health,
            commands_dir=out_path / "commands",
        )
        self.pattern.positions = schedule.positions
        self.slm_buffer.initialize(
            slm_shape,
            affine_transform,
            schedule.experiment_names,
            roi=camera_roi,
            grids=self.grids,
        )
        self.microscope.declare_slm()
        self.writer_process.base_path = out_path
        all_layers = self.writer_process.initialize(
            plan, self.core, affine_transform, slm_shape
        )

        self.all_layers = all_layers
        self.processes = [self.manager, self.slm_buffer, *router.active_processes()]

    def _instantiate_producers(self, schedule: ExperimentSchedule, router: Router):
        """Construct the methods of producer stages only where their output is demanded."""
        seg_demanded = sorted(
            {(exp, seg_name(kind)) for exp, _ch, kind in router.demanded_kinds("seg")}
        )
        for name, seg in seg_demanded:
            self.segmentation.request_method(schedule.experiments[name], seg)
        seg_demanded = set(seg_demanded)

        tracked = sorted(router.demanded("tracks"))
        for name, channel in tracked:
            self.tracking.request_method(schedule.experiments[name], channel)
        tracked_experiments = {name for name, _ in tracked}

        for name, experiment in schedule.experiments.items():
            for seg, cfg in experiment.segmentations.items():
                if cfg.method_name == "none" or (name, seg) in seg_demanded:
                    continue
                table = (
                    "[segmentation]"
                    if seg == DEFAULT_SEGMENTATION
                    else f"[segmentation.{seg}]"
                )
                which = "" if seg == DEFAULT_SEGMENTATION else f" {seg!r}"
                logger.warning(
                    f"experiment {name}: {table} method '{cfg.method_name}' is "
                    f"configured but the pattern method "
                    f"'{experiment.pattern.method_name}' does not request "
                    f"segmentation{which}, so it will not run"
                )
            method = experiment.tracking.method_name
            if name not in tracked_experiments and method != "none":
                logger.warning(
                    f"experiment {name}: tracking method '{method}' is configured "
                    f"but the pattern method '{experiment.pattern.method_name}' "
                    "does not request tracks, so no tracking will run"
                )

    def health(self) -> dict:
        """Process error counters, undeliverable frames and dropped frames, for status.json."""
        errors = {
            p.name: p.error_count
            for p in self.processes
            if hasattr(p, "error_count") and hasattr(p, "name")
        }
        return {
            "errors": errors,
            "undeliverable": self.router.undeliverable if self.router else 0,
            "dropped_frames": self.writer_process.dropped_frames,
        }

    def run(self):
        with ThreadPoolExecutor() as executor:
            # Map processes to futures
            future_to_process = {
                executor.submit(process.process): process for process in self.processes
            }

            # manager should be first to finish in a successful run
            manager_future = None
            for f, p in future_to_process.items():
                if p == self.manager:
                    manager_future = f
                    break

            try:
                # main process loop, checks for process exits
                while True:
                    # check if any process has exited (e.g., manager finishes, or crash)
                    done, _not_done = wait(
                        future_to_process.keys(),
                        return_when=FIRST_COMPLETED,
                        timeout=0.5,
                    )

                    """
                    Case 1: Manager first to finish
                    """
                    if manager_future in done:
                        exc = manager_future.exception()
                        if exc:
                            raise exc

                        logger.info(
                            "Manager finished successfully. Initiating graceful shutdown."
                        )
                        break

                    """
                    Case 2: Something else finished first
                    """
                    for f in done:
                        exc = f.exception()
                        if exc:
                            logger.error(
                                f"Process {future_to_process[f]} crashed with exception: {exc}"
                            )
                            raise exc

                        logger.warning(
                            f"Process {future_to_process[f]} exited unexpectedly (no exception)."
                        )

            except KeyboardInterrupt:
                logger.warning(
                    "KeyboardInterrupt caught in controller. Stopping all processes."
                )
                self.stop_event.set()

            except Exception as e:
                logger.error(f"Exception during run: {e}")
                logger.error(traceback.format_exc())
                self.stop_event.set()

            finally:
                logger.info("Waiting for all processes to exit...")

                try:
                    wait(future_to_process.keys(), return_when=ALL_COMPLETED)

                except KeyboardInterrupt:
                    logger.warning(
                        "Overriding stop_event handling, cancelling futures."
                    )
                    for f in future_to_process:
                        f.cancel()

                # the writer closes its own outputs when its loop exits; this covers
                # the case where the writer thread never ran
                # the last acknowledgements, the final status and the events table
                self.manager.finish()
                self.writer_process.close_files()
                self.all_queues.close()

                logger.info("Controller run finished.")
