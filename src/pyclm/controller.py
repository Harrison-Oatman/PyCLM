import logging
import traceback
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from threading import Event, active_count
from time import sleep

import numpy as np
from pymmcore_plus import CMMCorePlus

from .core import (
    ROI,
    AllQueues,
    CameraProperties,
    ExperimentSchedule,
    Manager,
    MicroscopeOutbox,
    MicroscopeProcess,
    PatternProcess,
    SegmentationProcess,
    SLMBuffer,
)
from .core.real_core import RealMicroscopeCore
from .core.virtual_microscope.simulated_core import SimulatedMicroscopeCore
from .core.virtual_microscope.simulated_source import TimeSeriesImageSource

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config="MMConfig_demo.cfg", dry=False):
        if not dry:
            # Applies if config specifies that a real microscope is in use
            self.core = RealMicroscopeCore()
        else:
            image_source = TimeSeriesImageSource(Path("tif-source"), loop=True)
            self.core = SimulatedMicroscopeCore(image_source, slm_device=None)
        self.core.loadSystemConfiguration(config)
        self.all_queues = AllQueues()

        self.stop_event = Event()

        self.microscope = MicroscopeProcess(
            core=self.core, aq=self.all_queues, stop_event=self.stop_event
        )
        self.manager = Manager(aq=self.all_queues, stop_event=self.stop_event)
        self.outbox = MicroscopeOutbox(aq=self.all_queues, stop_event=self.stop_event)
        self.slm_buffer = SLMBuffer(aq=self.all_queues, stop_event=self.stop_event)
        self.segmentation = SegmentationProcess(
            aq=self.all_queues, stop_event=self.stop_event
        )
        self.pattern = PatternProcess(aq=self.all_queues, stop_event=self.stop_event)

        self.processes = [
            self.microscope,
            self.manager,
            self.outbox,
            self.slm_buffer,
            self.segmentation,
            self.pattern,
        ]

        self.camera_properties = None

    def set_binning(self, binning: int):
        core = self.core
        camera = self.core.getCameraDevice()

        try:
            allowed = core.getAllowedPropertyValues(camera, "Binning")
        except:
            return None

        binning_str = f"{binning}x{binning}"

        if binning_str in allowed:
            core.setProperty(camera, "Binning", binning_str)

        else:
            logger.warning(
                f"attempted set binning {binning_str}, allowed binnings {allowed}"
            )

    def register_pattern_method(self, name: str, method: type):
        self.pattern.register_method(method, name)

    def register_segmentation_method(self, name: str, method: type):
        self.segmentation.register_method(method, name)

    def initialize(
        self,
        schedule: ExperimentSchedule,
        slm_shape: tuple[int, int],
        affine_transform: np.ndarray,
        out_path: Path,
    ):
        self.set_binning(1)

        camera_roi = ROI(*self.core.getROI())
        camera_resolution = self.core.getPixelSizeUm()

        self.camera_properties = CameraProperties(camera_roi, camera_resolution)

        logger.info(f"camera properties: {self.camera_properties}")

        self.pattern.initialize(self.camera_properties)

        pattern_requirements = {}
        for name, experiment in schedule.experiments.items():
            pattern_requirements[name] = self.pattern.request_method(experiment)

            if any([req.needs_seg for req in pattern_requirements[name]]):
                self.segmentation.request_method(experiment)

        self.pattern.initialize_models()

        self.manager.initialize(schedule, pattern_requirements)
        self.slm_buffer.initialize(
            slm_shape, affine_transform, schedule.experiment_names
        )
        self.microscope.declare_slm()
        self.outbox.base_path = out_path
        self.outbox.initialize(schedule)

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

                self.all_queues.close()

                logger.info("Controller run finished.")
