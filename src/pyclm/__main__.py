from pathlib import Path
from argparse import ArgumentParser

from . import run_pyclm

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
    print("running pyclm...")
    main()