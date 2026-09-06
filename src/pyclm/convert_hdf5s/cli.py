import logging
from argparse import ArgumentParser
from pathlib import Path

from natsort import natsorted
from tqdm import tqdm

from .affine import AffineCalibration
from .pipeline import convert_file

logger = logging.getLogger(__name__)


def process_args():
    parser = ArgumentParser()
    parser.add_argument("directory", help="directory containing experiment files")
    parser.add_argument("channels", nargs="*", help="channels to extract")
    parser.add_argument(
        "--config", type=str, help="path to pyclm_config.toml file", default=None
    )
    parser.add_argument(
        "--binning",
        type=int,
        help="binning during experiment (autodetected if not specified)",
        default=None,
    )
    parser.add_argument(
        "--roi",
        type=str,
        help="camera ROI offset override 'x_offset,y_offset' "
        "(autodetected from file metadata if not specified)",
        default=None,
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable caching of repeated DMD pattern warps",
    )

    return parser.parse_args()


def find_config(input_dir: Path, config_path: str | None) -> Path:
    if config_path is None:
        # look in the experiment directory for pyclm_config.toml
        config_path = input_dir / "pyclm_config.toml"

        # look in the current working directory for pyclm_config.toml
        if not config_path.exists():
            config_path = Path("pyclm_config.toml")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. Affine transform is required."
        )

    return config_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = process_args()
    input_dir = Path(args.directory)

    config_path = find_config(input_dir, args.config)
    calibration = AffineCalibration.from_config(
        config_path, cache_size=0 if args.no_cache else 2
    )

    roi_override = None
    if args.roi:
        x_offset, y_offset = (int(v) for v in args.roi.split(","))
        roi_override = (x_offset, y_offset)

    for path in tqdm(natsorted(input_dir.glob("*.hdf5"))):
        convert_file(
            path,
            args.channels,
            calibration,
            binning_override=args.binning,
            roi_override=roi_override,
        )


if __name__ == "__main__":
    main()
