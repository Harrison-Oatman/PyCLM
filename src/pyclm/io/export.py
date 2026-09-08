"""
Export PyCLM experiments to ImageJ hyperstacks: one TIFF per cadence group,
channels = raw channels, then segmentation labels (if any), then tracked
labels (if any), then the DMD pattern warped into camera space (if the
affine transform is known).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import tifffile
from skimage.transform import downscale_local_mean

from .readers import ExperimentData, GroupData

logger = logging.getLogger(__name__)


def _lut(r, g, b):
    ramp = np.arange(256, dtype=np.uint8)
    zero = np.zeros(256, dtype=np.uint8)
    return np.stack([ramp if r else zero, ramp if g else zero, ramp if b else zero])


GRAY, YELLOW, CYAN, MAGENTA = _lut(1, 1, 1), _lut(1, 1, 0), _lut(0, 1, 1), _lut(1, 0, 1)
GREEN, RED, BLUE = _lut(0, 1, 0), _lut(1, 0, 0), _lut(0, 0, 1)
# one LUT per segmentation table, in the store's order (the default first)
LABEL_LUTS = (YELLOW, GREEN, RED, BLUE)


def pattern_to_camera(
    pattern: np.ndarray, affine: np.ndarray, shape: tuple[int, int], binning: int
) -> np.ndarray:
    """Warp a DMD-space pattern back into a (binned) camera frame, values 0-255."""
    h, w = shape
    ati = cv2.invertAffineTransform(np.asarray(affine, dtype=np.float32))
    full = cv2.warpAffine(
        np.round(pattern).astype(np.uint8), ati, (w * binning, h * binning)
    )
    if binning > 1:
        full = downscale_local_mean(full.astype(np.float32), (binning, binning))
    return np.clip(full, 0, 255).astype(np.uint16)


def export_group(
    exp: ExperimentData,
    group: GroupData,
    out_path: Path,
    affine: np.ndarray | None = None,
    overlay_pattern: bool = True,
    include_labels: bool = True,
) -> Path | None:
    affine = exp.affine_transform if affine is None else affine
    indices = group.acquired()
    if not indices:
        logger.warning(f"{exp.name}/{group.name}: nothing acquired, no export")
        return None

    shape = group.shape
    label_names = list(group.label_names) if include_labels else []
    use_tracks = include_labels and group.has_tracks
    use_pattern = (
        overlay_pattern
        and affine is not None
        and exp.pattern_at(group.global_t(indices[0])) is not None
    )

    frames = []
    for i in indices:
        planes = []
        for c in group.channels:
            fr = group.frame(i, c)
            planes.append(
                np.zeros(shape, np.uint16) if fr is None else fr.astype(np.uint16)
            )
        for name in label_names:
            for c in group.channels:
                lb = group.labels(i, c, name)
                planes.append(
                    np.zeros(shape, np.uint16) if lb is None else lb.astype(np.uint16)
                )
        if use_tracks:
            for c in group.channels:
                tr = group.tracks(i, c)
                if tr is None:
                    planes.append(np.zeros(shape, np.uint16))
                else:
                    if tr.max() > np.iinfo(np.uint16).max:
                        logger.warning(
                            f"{exp.name}/{group.name}: track ids exceed 65535 and are "
                            "clipped in the ImageJ export; read the zarr store for exact ids"
                        )
                    planes.append(
                        np.clip(tr, 0, np.iinfo(np.uint16).max).astype(np.uint16)
                    )
        if use_pattern:
            pat = exp.pattern_at(group.global_t(i))
            planes.append(
                np.zeros(shape, np.uint16)
                if pat is None
                else pattern_to_camera(pat, affine, shape, group.binning)
            )
        frames.append(np.stack(planes))

    stack = np.stack(frames)  # (T, C, Y, X)
    luts = [GRAY] * len(group.channels)
    ranges = [0, 5000] * len(group.channels)
    for k, _name in enumerate(label_names):
        luts += [LABEL_LUTS[k % len(LABEL_LUTS)]] * len(group.channels)
        ranges += [0, 1] * len(group.channels)
    if use_tracks:
        luts += [MAGENTA] * len(group.channels)
        ranges += [0, 1] * len(group.channels)
    if use_pattern:
        luts.append(CYAN)
        ranges += [0, 1000]

    metadata = {
        "axes": "TCYX",
        "mode": "composite",
        "LUTs": luts,
        "Ranges": ranges,
    }
    kwargs = {}
    if group.interval_s:
        metadata["finterval"] = float(
            group.every_t * group.interval_s
        )  # seconds per frame
    if group.pixel_size_um:
        metadata["unit"] = "um"
        kwargs["resolution"] = (1.0 / group.pixel_size_um, 1.0 / group.pixel_size_um)

    out_path = Path(out_path)
    tifffile.imwrite(out_path, stack, imagej=True, metadata=metadata, **kwargs)
    logger.info(f"exported {exp.name}/{group.name} -> {out_path} {stack.shape}")
    return out_path


def export_imagej(
    exp: ExperimentData,
    out_dir: Path | None = None,
    groups: list[str] | None = None,
    affine: np.ndarray | None = None,
    overlay_pattern: bool = True,
    include_labels: bool = True,
) -> list[Path]:
    """Write one ImageJ hyperstack per cadence group; returns the paths written."""
    out_dir = Path(out_dir) if out_dir is not None else exp.path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, group in exp.groups.items():
        if groups is not None and name not in groups:
            continue
        path = out_dir / f"{exp.name}_{name}.tif"
        result = export_group(exp, group, path, affine, overlay_pattern, include_labels)
        if result is not None:
            written.append(result)
    return written
