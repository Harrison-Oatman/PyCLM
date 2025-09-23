from argparse import ArgumentParser
import logging
from toml import load
from typing import Optional

import numpy as np

from pathlib import Path

from src import Controller, SegmentationMethod, PatternMethod, schedule_from_directory




def set_logging(experiment_directory: Path):
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(experiment_directory / 'log.log')

    # Set levels for handlers
    console_handler.setLevel(logging.WARNING)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(filename)-12s \t - %(levelname)-8s - %(message)s',
                                    datefmt='%H:%M:%S')

    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    logging.basicConfig(handlers=[console_handler, file_handler])





def main():
    args = process_args()
    base_path = Path(args.directory)
    run_pyclm(base_path, args.config)


def process_args():
    parser = ArgumentParser()
    parser.add_argument("directory", help="directory containing experiment files")
    parser.add_argument("--config", type=str, help="path to pyclm_config.toml file", default=None)

    return parser.parse_args()


if __name__ == '__main__':
    main()
