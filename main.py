from argparse import ArgumentParser
import logging
from src import *
from pymmcore_plus import CMMCorePlus
# from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


class Controller:

    def __init__(self):
        self.core = CMMCorePlus()
        self.core.loadSystemConfiguration()
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

    def run(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process.process) for process in self.processes]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Exception occurred: {e}")


def main():
    args = process_args()

    c = Controller()

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
