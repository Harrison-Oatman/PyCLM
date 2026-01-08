import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pymmcore_plus import CMMCorePlus

import numpy as np

from .core import AllQueues, MicroscopeProcess, Manager, MicroscopeOutbox, SLMBuffer, SegmentationProcess, \
    PatternProcess, ExperimentSchedule, ROI, CameraProperties

from .core.real_core import RealMicroscopeCore
from .core.virtual_microscope.simulated_core import SimulatedMicroscopeCore
from .core.virtual_microscope.simulated_source import TimeSeriesImageSource

logger = logging.getLogger(__name__)


class Controller:

    def __init__(self, config="MMConfig_demo.cfg", dry=False):
        if (not dry):
            # Applies if config specifies that a real microscope is in use
            self.core = RealMicroscopeCore()
        else:
            # image_source = TimeSeriesImageSource.from_tiff_stack(Path("path/to/stack.tif"), loop=True)
            image_source = TimeSeriesImageSource.from_folder(Path("tif-source"), pattern="*.tif", loop=True)
            self.core = SimulatedMicroscopeCore(image_source, slm_device=None)
        self.core.loadSystemConfiguration(config)
        self.all_queues = AllQueues()

        self.microscope = MicroscopeProcess(core=self.core, aq=self.all_queues)
        self.manager = Manager(aq=self.all_queues)
        self.outbox = MicroscopeOutbox(aq=self.all_queues, save_type="hdf5")
        self.slm_buffer = SLMBuffer(aq=self.all_queues)
        self.segmentation = SegmentationProcess(aq=self.all_queues)
        self.pattern = PatternProcess(aq=self.all_queues)

        self.processes = [
            self.microscope,
            self.manager,
            self.outbox,
            self.slm_buffer,
            self.segmentation,
            self.pattern
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
            logger.warning(f"attempted set binning {binning_str}, allowed binnings {allowed}")

    def register_pattern_method(self, name: str, method: type):
        self.pattern.register_method(method, name)

    def register_segmentation_method(self, name: str, method: type):
        self.segmentation.register_method(method, name)

    def initialize(self, schedule: ExperimentSchedule, slm_shape: tuple[int, int], affine_transform: np.ndarray,
                   out_path: Path):

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

        self.manager.initialize(schedule, pattern_requirements)
        self.slm_buffer.initialize(slm_shape, affine_transform, schedule.experiment_names)
        self.microscope.declare_slm()
        self.outbox.base_path = out_path

    def run(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process.process) for process in self.processes]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Exception occurred: {e}")
                    logging.error(traceback.format_exc())
                # future.result()
