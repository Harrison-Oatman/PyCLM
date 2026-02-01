import json
import re
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import tifffile
from h5py import File
from natsort import natsorted
from skimage.transform import downscale_local_mean
from toml import load
from tqdm import tqdm


def get_mapping(projector_api):
    mapping = projector_api.load_mapping(projector_api.get_projection_device())

    m = mapping.get_map().values()

    # horrible regex, TODO use ast instead
    pattern = r"AffineTransform\[\[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\], \[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]\]"

    # Find all matches
    matches = re.findall(pattern, m.to_string())

    M = np.stack(np.array(matches).astype(float)).reshape(-1, 2, 3)
    at = np.median(M, axis=0)

    print(f"Affine Transform from calibration: {at}")
    return at


def get_binning_from_metadata(f: File, chan_key: str):
    """
    Extract binning from file attributes or return default 1.
    """
    if "experiment_metadata" in f.attrs:
        try:
            meta = json.loads(f.attrs["experiment_metadata"])
            # chan_key is typically "channel_NAME"
            # channel keys in metadata are "NAME"
            if chan_key.startswith("channel_"):
                short_name = chan_key.replace("channel_", "", 1)
                if short_name in meta.get("channels", {}):
                    return meta["channels"][short_name].get("binning", 1)
        except Exception as e:
            print(f"Error reading binning from metadata: {e}")

    return 1


def make_tif(fp, at, chan="channel_638"):
    ati = cv2.invertAffineTransform(at)
    patterned = []

    collected_frames = []
    channel_key = f"{chan}"

    # Open with SWMR support
    try:
        f = File(fp, mode="r", libver="latest", swmr=True)
    except OSError:
        # Fallback if file not accessible or not HDF5
        print(f"Could not open {fp}")
        return

    # Attempt to read binning from metadata
    binning = get_binning_from_metadata(f, channel_key)

    try:
        indices = []

        keys = list(f.keys())
        for t_val in keys:
            if t_val not in f:
                continue
            data_group = f[t_val]
            if channel_key not in data_group:
                continue

            indices.append(t_val)

        seg_seen = False

        for t_val in natsorted(indices):
            # Check if dataset exists / is complete
            try:
                if channel_key not in f[t_val] or "data" not in f[t_val][channel_key]:
                    continue

                data_dset = f[t_val][channel_key]["data"]
                # Ensure data is accessible (SWMR safety)
                data_dset.refresh()
                data = np.array(data_dset)

            except Exception as e:
                # might happen if writing is in progress for this specific frame
                continue

            collected_frames.append(data)

            patterned_stack = [data]

            if "seg" in f[t_val][channel_key].keys():
                patterned_stack.append(f[t_val][channel_key]["seg"])
                seg_seen = True

            elif seg_seen:
                continue

            if "stim_aq" in f[t_val].keys() and "dmd" in f[t_val]["stim_aq"]:
                pattern = np.array(f[t_val]["stim_aq"]["dmd"])
                target_size = data.shape

                # Ensure binning is valid integer
                b = int(binning)

                tf = cv2.warpAffine(
                    np.round(pattern).astype(np.uint8),
                    ati,
                    (target_size[1] * b, target_size[0] * b),
                ).astype(np.uint16)
                ds = downscale_local_mean(tf, (b, b)).astype(np.uint16)
                patterned_stack.append(ds)

            else:
                patterned_stack.append(np.zeros(data.shape))

            patterned.append(np.stack(patterned_stack).astype(np.uint16))

    finally:
        f.close()

    if not collected_frames:
        return

    # Save patterned output
    if patterned:
        # Construct output filename
        outpath_pattern = str(fp)[:-5] + f"_{chan}_patterns.tif"
        tifffile.imwrite(
            outpath_pattern,
            np.array(patterned).astype(np.uint16),
            imagej=True,
            metadata={"axes": "tcyx"},
        )
        print(f"Saved {outpath_pattern}")


def process_args():
    parser = ArgumentParser()
    parser.add_argument("directory", help="directory containing experiment files")
    parser.add_argument("channels", nargs="*", help="channels to extract")
    parser.add_argument(
        "--config", type=str, help="path to pyclm_config.toml file", default=None
    )
    # Removed binning, overlay_pattern, just_patterns args

    return parser.parse_args()


def find_affine_transform(input_dir, config_path):
    # copied from main.py
    # search for config file if not provided
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

    config = load(config_path)
    return np.array(config["affine_transform"], dtype=np.float32)


def main():
    args = process_args()
    input_dir = args.directory
    config_path = args.config
    channels = args.channels

    # We require overlay pattern approach, so we need affine transform
    at = find_affine_transform(Path(input_dir), config_path)

    for val in tqdm(list(Path(input_dir).glob("*.hdf5"))):
        for c in channels:
            make_tif(str(val), at, f"channel_{c}")


if __name__ == "__main__":
    main()
