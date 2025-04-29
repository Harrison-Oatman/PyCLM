from argparse import ArgumentParser
import logging
import traceback

import numpy as np

from src import *
from pymmcore_plus import CMMCorePlus
# from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


class Controller:

    def __init__(self, config="MMConfig_demo.cfg"):
        self.core = CMMCorePlus()
        self.core.loadSystemConfiguration(config)
        self.all_queues = AllQueues()

        self.microscope = MicroscopeProcess(core=self.core, aq=self.all_queues)
        self.manager = Manager(aq=self.all_queues)
        self.outbox = MicroscopeOutbox(aq=self.all_queues)
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

    def initialize(self, schedule: ExperimentSchedule, slm_shape: tuple[int, int], affine_transform: np.ndarray,
                   out_path: Path):

        camera_roi = ROI(*self.core.getROI())
        camera_resolution = self.core.getPixelSizeUm()

        self.camera_properties = CameraProperties(camera_roi, camera_resolution)

        self.pattern.initialize(self.camera_properties)

        pattern_requirements = {}
        for name, experiment in schedule.experiments.items():
            pattern_requirements[name] = self.pattern.request_model(experiment)

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

def get_slm_shape(core: CMMCorePlus):

    dev = core.getSLMDevice()
    return core.getSLMHeight(dev), core.getSLMWidth(dev)


# def main():
#     args = process_args()
#
#     c = Controller(r"C:\Program Files\Micro-Manager-2.0\Ti2MightexCrestSolaSpectra.cfg")
#
#     core = c.core
#     for group in core.getAvailableConfigGroups():
#         cg = core.getConfigGroupObject(group, False)
#         print(cg.name, list(cg.items()))
#
#     base_path = Path(str(r"C:\Users\Nikon\Desktop\Code\FeedbackControl\test\test1"))
#     schedule = schedule_from_directory(base_path)
#
#     slm_shape = get_slm_shape(core)
#
#     core.setFocusDevice("ZDrive")
#
#     at = np.array([[-.289, 0.006, 959.025], [-0.012, -0.579, 1540.03]], dtype=np.float32)
#     c.initialize(schedule, slm_shape, at, base_path)
#     logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
#
#     c.run()


def main():
    args = process_args()

    c = Controller()

    core = c.core
    for group in core.getAvailableConfigGroups():
        cg = core.getConfigGroupObject(group, False)
        print(list(cg.items()))

    t_interval = 5
    t_steps = 5
    t_setup = 2
    t_between = 1
    e1_toml_path = r"D:\FeedbackControl\experiments\SampleExperiment.toml"
    e2_toml_path = r"D:\FeedbackControl\experiments\SampleExperiment2.toml"
    experiments = {"a": experiment_from_toml(e1_toml_path, "a"),
                   "b": experiment_from_toml(e2_toml_path, "b")}
    positions = {
        "a": Position(1, 2, 0),
        "b": Position(3, 4, 0)
    }

    schedule = ExperimentSchedule(experiments, positions, t_steps, t_interval, t_setup, t_between)
    slm_shape = (512, 512)
    at = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    base_path = Path(r"D:\FeedbackControl\test")

    c.initialize(schedule, slm_shape, at, base_path)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    c.run()


def process_args():
    parser = ArgumentParser()
    # parser.add_argument("--config", type=str, help="path to config file")
    # parser.add_argument("--log", type=str, help="path to log file")
    parser.add_argument("--debug", action="store_true", help="enable debug logging")

    return parser.parse_args()


if __name__ == '__main__':
    main()
