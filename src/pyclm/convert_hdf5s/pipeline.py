import logging
from pathlib import Path

import numpy as np

from .affine import AffineCalibration
from .hdf5_source import ExperimentFile
from .tiff_writer import write_pattern_tif

logger = logging.getLogger(__name__)


def convert_channel(
    path: Path,
    channel_key: str,
    calibration: AffineCalibration,
    binning_override: int | None = None,
    roi_override: tuple[int, int] | None = None,
) -> Path | None:
    """
    Convert a single channel of a single experiment HDF5 file into a
    `<stem>_<channel_key>_patterns.tif` stack, overlaying the DMD
    stimulation pattern (warped into camera coordinates) on each frame.

    Returns the output path, or None if the channel has no data.
    """
    with ExperimentFile(path) as ef:
        binning = binning_override or ef.read_binning(channel_key)
        roi = roi_override if roi_override is not None else ef.read_camera_roi()
        has_seg = ef.has_seg(channel_key)

        frames = []
        for _t_key, data, seg, dmd in ef.iter_frames(channel_key):
            if dmd is not None:
                pattern = calibration.warp_pattern_to_camera(
                    dmd, data.shape, binning=binning, roi=roi
                )
            else:
                pattern = np.zeros(data.shape, dtype=np.uint16)

            stack = [data, seg, pattern] if has_seg else [data, pattern]
            frames.append(np.stack(stack))

    if not frames:
        return None

    out_path = path.with_name(f"{path.stem}_{channel_key}_patterns.tif")
    write_pattern_tif(out_path, frames, has_seg)
    return out_path


def convert_file(
    path: Path,
    channels: list[str],
    calibration: AffineCalibration,
    binning_override: int | None = None,
    roi_override: tuple[int, int] | None = None,
) -> None:
    for c in channels:
        channel_key = "stim_aq" if c == "stim" else f"channel_{c}"
        out_path = convert_channel(
            path, channel_key, calibration, binning_override, roi_override
        )
        if out_path is not None:
            logger.info("Saved %s", out_path)
