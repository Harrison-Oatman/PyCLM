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


# def apply_at(img, at, target_size):
#
#
#


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


    outpath = fp[:-5] + ".tif"
    tifffile.imwrite(outpath, np.array(collected_frames), imagej=True, metadata={"axes": "tyx"})

    if at is not None:
        # print(np.array(patterned).shape)
        tifffile.imwrite(fp[:-5] + "_patterns.tif", np.array(patterned).astype(np.uint16), imagej=True, metadata={"axes": "tcyx"})


def parse_args():
    args = ArgumentParser()

    args.add_argument("dir")

    return args.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # input_dir = args.dir

    input_dir = r"E:\Harrison\RTx3 imaging\2025-08-21 tag-rfp 1\global cycle\logofinal"
    at = np.array([[-.289, 0.006, 959.025], [-0.012, -0.579, 1540.03]], dtype=np.float32)

    for val in tqdm(Path(input_dir).glob("*.hdf5")):
        make_tif(str(val), at, "channel_545")

    # make_tif(r"D:\FeedbackControl\bar5.08.hdf5")
