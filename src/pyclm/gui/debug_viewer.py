import os

os.environ.setdefault("NAPARI_DISABLE_PLUGIN_AUTOLOAD", "1")

import sys

import napari
import numpy as np
from h5py import File


def view_hdf5(filepath: str, channel="channel_638"):
    viewer = napari.Viewer()
    with File(filepath, mode="r", libver="latest", swmr=True) as f:
        timepoints = sorted(f.keys())
        print("Timepoints:", timepoints)

        stack = []
        for t in timepoints:
            if channel in f[t] and "data" in f[t][channel]:
                data = np.array(f[t][channel]["data"])
                stack.append(data)

        if not stack:
            raise RuntimeError(f"No data found for channel {channel}")

        stack = np.stack(stack)
        print("Stack shape:", stack.shape)
        print("dtype/min/max:", stack.dtype, stack.min(), stack.max())

        viewer.add_image(stack, name=channel)

    napari.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    filepath = sys.argv[1]
    channel = sys.argv[2] if len(sys.argv) > 2 else "channel_638"
    view_hdf5(filepath, channel)
