from pathlib import Path

import numpy as np
import tifffile

from .imagej_meta import build_metadata


def write_pattern_tif(out_path: Path, frames: list[np.ndarray], has_seg: bool) -> None:
    """
    Write a tcyx ImageJ-readable TIFF stack.

    `frames` is a list of (C, H, W) arrays, one per timepoint, each ordered
    as [data, (seg,) pattern].
    """
    if not frames:
        return

    stack = np.stack(frames).astype(np.uint16)
    metadata = build_metadata(has_seg)

    tifffile.imwrite(out_path, stack, imagej=True, metadata=metadata)
