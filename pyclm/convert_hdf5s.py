import re
import numpy as np
from h5py import File
import tifffile
from pathlib import Path
from argparse import ArgumentParser
from natsort import natsorted
from skimage.transform import downscale_local_mean
import cv2
from tqdm import tqdm
from toml import load


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


def extract_channels_tifs(fp, chans):
    for chan in chans:
        collected_frames = []
        channel_key = f"{chan}"

        with (File(fp, mode="r") as f):

            indices = []

            for t_val, data in f.items():

                if channel_key not in data:
                    continue

                indices.append(t_val)

            for t_val in natsorted(indices):

                data = np.array(f[t_val][channel_key]["data"])
                collected_frames.append(data)

        outpath = f"{fp[:-5]}_{chan}.tif"
        tifffile.imwrite(outpath, np.array(collected_frames), imagej=True, metadata={"axes": "tyx"})


def make_tif(fp, at=None, chan="channel_638"):

    if at is not None:
        ati = cv2.invertAffineTransform(at)
        patterned = []

    collected_frames = []
    channel_key = f"{chan}"

    with (File(fp, mode="r") as f):

        indices = []

        for t_val, data in f.items():

            if channel_key not in data:
                continue

            indices.append(t_val)

        seg_seen = False

        for t_val in natsorted(indices):

            data = np.array(f[t_val][channel_key]["data"])
            collected_frames.append(data)

            patterned_stack = [data]

            if at is not None:

                if "seg" in f[t_val][channel_key].keys():

                    patterned_stack.append(f[t_val][channel_key]["seg"])

                    seg_seen = True

                elif seg_seen:
                    continue

                if "stim_aq" in f[t_val].keys():

                    pattern = np.array(f[t_val]["stim_aq"]["dmd"])
                    target_size = data.shape
                    tf = cv2.warpAffine(np.round(pattern).astype(np.uint8), ati, (target_size[1]*2, target_size[0]*2)).astype(np.uint16)
                    ds = downscale_local_mean(tf, (2, 2)).astype(np.uint16)
                    patterned_stack.append(ds)

                else:
                    patterned_stack.append(np.zeros(data.shape))

                patterned.append(np.stack(patterned_stack).astype(np.uint16))


    outpath = fp[:-5] + chan + ".tif"
    tifffile.imwrite(outpath, np.array(collected_frames), imagej=True, metadata={"axes": "tyx"})

    if at is not None:
        # print(np.array(patterned).shape)
        tifffile.imwrite(fp[:-5] + chan + "_patterns.tif", np.array(patterned).astype(np.uint16), imagej=True, metadata={"axes": "tcyx"})

def make_stim_tif(fp, at):
    print(fp)

    ati = cv2.invertAffineTransform(at)
    patterned = []

    with (File(fp, mode="r") as f):

        indices = []

        for t_val, data in f.items():

            if "stim_aq" not in data:
                continue

            indices.append(t_val)

        for t_val in natsorted(indices):
            pattern = np.array(f[t_val]["stim_aq"]["dmd"])
            target_size = (1600, 1600)
            tf = cv2.warpAffine(np.round(pattern).astype(np.uint8), ati,
                                (target_size[1] * 2, target_size[0] * 2)).astype(np.uint16)
            ds = downscale_local_mean(tf, (2, 2)).astype(np.uint16)
            patterned.append(ds)

    tifffile.imwrite(str(fp)[:-5] + "patterns_only.tif", np.array(patterned).astype(np.uint16), imagej=True, metadata={"axes": "tyx"})


def process_args():
    parser = ArgumentParser()
    parser.add_argument("directory", help="directory containing experiment files")
    parser.add_argument("channels", nargs='*', help="channels to extract")
    parser.add_argument("--config", type=str, help="path to pyclm_config.toml file", default=None)
    parser.add_argument("--overlay_pattern", action="store_true", help="whether to overlay the pattern on the tif")
    parser.add_argument("--just_patterns", action="store_true", help="just add the stimulation")

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

    assert input_dir.exists(), f"experiment directory {input_dir} does not exist"
    assert config_path.exists(), (
        f"config file {config_path} does not exist: pyclm_config.toml must be specified or be "
        f"present in the experiment directory")

    config = load(config_path)
    return np.array(config["affine_transform"], dtype=np.float32)


def main():
    args = process_args()
    input_dir = args.directory
    config_path = args.config
    channels = args.channels
    overlay_pattern = args.overlay_pattern

    if args.just_patterns:
        at = find_affine_transform(Path(input_dir), config_path)
        for val in tqdm(Path(input_dir).glob("*.hdf5")):
            make_stim_tif(val, at)

        return 0

    if overlay_pattern:
        at = find_affine_transform(Path(input_dir), config_path)
    else:
        at = None

    for val in tqdm(Path(input_dir).glob("*.hdf5")):
        for c in channels:
            if overlay_pattern:
                make_tif(str(val), at, "channel_545")

            else:
                extract_channels_tifs(str(val), [f"channel_{c}"])

if __name__ == "__main__":
    main()
