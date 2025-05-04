import re
import numpy as np
from h5py import File
import tifffile
from pathlib import Path
from argparse import ArgumentParser
from natsort import natsorted


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


def make_tif(fp, chan="638"):

    collected_frames = []
    channel_key = f"channel_{chan}"

    with File(fp, mode="r") as f:

        indices = []

        for t_val, data in f.items():

            if channel_key not in data:
                continue

            indices.append(t_val)

        for t_val in natsorted(indices):

            collected_frames.append(np.array(f[t_val][channel_key]["data"]))

    outpath = fp[:-5] + ".tif"
    tifffile.imwrite(outpath, np.array(collected_frames))


def parse_args():
    args = ArgumentParser()

    args.add_argument("dir")

    return args.parse_args()


if __name__ == "__main__":

    args = parse_args()

    input_dir = args.dir

    for val in Path(input_dir).glob("*.hdf5"):
        make_tif(str(val))

    # make_tif(r"D:\FeedbackControl\bar5.08.hdf5")
