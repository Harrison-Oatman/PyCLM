from unittest import mock

import cv2
import h5py
import numpy as np
import pytest
import tifffile
from skimage.transform import downscale_local_mean

from pyclm.convert_hdf5s import AffineCalibration, ExperimentFile, convert_channel


def _legacy_warp(pattern, at, out_shape, binning):
    """Reference implementation: upscale-then-downsample, as before the refactor."""
    ati = cv2.invertAffineTransform(at.astype(np.float32))
    h, w = out_shape
    tf = cv2.warpAffine(
        np.round(pattern).astype(np.uint8), ati, (w * binning, h * binning)
    ).astype(np.uint16)
    return downscale_local_mean(tf, (binning, binning)).astype(np.uint16)


@pytest.mark.parametrize("binning", [1, 2, 4])
def test_warp_matches_legacy_downsample(binning):
    # A linear ramp's box-average over a block equals its value at the
    # block's center, so the downsample-folded warp should closely match
    # the old warpAffine + downscale_local_mean result.
    size = 128
    ramp = np.linspace(0, 255, size)
    pattern = np.tile(ramp, (size, 1)).astype(np.uint8)

    at = np.array([[0.6, 0.05, 5.0], [-0.04, 0.6, 8.0]], dtype=np.float64)
    out_shape = (24, 24)

    legacy = _legacy_warp(pattern, at, out_shape, binning)

    calibration = AffineCalibration(at)
    new = calibration.warp_pattern_to_camera(pattern, out_shape, binning=binning)

    assert new.shape == legacy.shape == out_shape
    assert np.mean(np.abs(legacy.astype(np.int32) - new.astype(np.int32))) < 1.0


def test_roi_offset_with_identity_transform():
    pattern = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 256).astype(
        np.uint8
    )

    at = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    calibration = AffineCalibration(at)
    out_shape = (16, 16)

    no_roi = calibration.warp_pattern_to_camera(pattern, out_shape, binning=1, roi=None)
    zero_roi = calibration.warp_pattern_to_camera(
        pattern, out_shape, binning=1, roi=(0, 0, 64, 64)
    )
    assert np.array_equal(no_roi, zero_roi)
    assert np.array_equal(no_roi, pattern[:16, :16])

    shifted = calibration.warp_pattern_to_camera(
        pattern, out_shape, binning=1, roi=(8, 4, 64, 64)
    )
    assert np.array_equal(shifted, pattern[4:20, 8:24])


def test_pattern_cache_avoids_redundant_warps():
    rng = np.random.default_rng(0)
    shape = (32, 32)
    pattern_a = rng.integers(0, 256, size=shape, dtype=np.uint8)
    pattern_b = rng.integers(0, 256, size=shape, dtype=np.uint8)
    pattern_c = rng.integers(0, 256, size=shape, dtype=np.uint8)

    at = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    calibration = AffineCalibration(at, cache_size=2)
    out_shape = (16, 16)

    with mock.patch("cv2.warpAffine", wraps=cv2.warpAffine) as mock_warp:
        calibration.warp_pattern_to_camera(pattern_a, out_shape)  # miss -> 1
        calibration.warp_pattern_to_camera(pattern_a, out_shape)  # hit
        calibration.warp_pattern_to_camera(pattern_b, out_shape)  # miss -> 2
        calibration.warp_pattern_to_camera(pattern_a, out_shape)  # hit (still cached)
        calibration.warp_pattern_to_camera(pattern_c, out_shape)  # miss -> 3, evicts a
        calibration.warp_pattern_to_camera(pattern_b, out_shape)  # hit
        calibration.warp_pattern_to_camera(pattern_a, out_shape)  # miss (evicted) -> 4

    assert mock_warp.call_count == 4


def test_iter_frames_consistent_stack_depth(tmp_path):
    path = tmp_path / "exp.hdf5"

    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(
            "00000/channel_638/data", data=np.full((8, 8), 10, dtype=np.uint16)
        )
        f.create_dataset(
            "00000/channel_638/seg", data=np.full((8, 8), 1, dtype=np.uint16)
        )
        f.create_dataset(
            "00000/stim_aq/dmd", data=np.full((32, 32), 255, dtype=np.uint8)
        )

        f.create_dataset(
            "00001/channel_638/data", data=np.full((8, 8), 20, dtype=np.uint16)
        )
        # t=1 has no seg and no stim pattern

    with ExperimentFile(path) as ef:
        assert ef.read_camera_roi() is None
        assert ef.has_seg("channel_638") is True

        frames = list(ef.iter_frames("channel_638"))

    assert [t_key for t_key, *_ in frames] == ["00000", "00001"]

    _, data0, seg0, dmd0 = frames[0]
    _, data1, seg1, dmd1 = frames[1]

    assert seg0 is not None
    assert seg0.shape == data0.shape
    assert seg1 is not None
    assert seg1.shape == data1.shape
    assert np.all(seg1 == 0)

    assert dmd0 is not None
    assert dmd1 is None


def test_convert_channel_writes_consistent_stack(tmp_path):
    path = tmp_path / "exp.hdf5"

    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(
            "00000/channel_638/data", data=np.full((8, 8), 10, dtype=np.uint16)
        )
        f.create_dataset(
            "00000/channel_638/seg", data=np.full((8, 8), 1, dtype=np.uint16)
        )
        f.create_dataset(
            "00000/stim_aq/dmd", data=np.full((32, 32), 255, dtype=np.uint8)
        )

        f.create_dataset(
            "00001/channel_638/data", data=np.full((8, 8), 20, dtype=np.uint16)
        )

    at = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    calibration = AffineCalibration(at)

    out_path = convert_channel(path, "channel_638", calibration)

    assert out_path is not None
    stack = tifffile.imread(out_path)

    assert stack.shape == (2, 3, 8, 8)  # (t, [data, seg, pattern], h, w)
    assert np.all(stack[1, 1] == 0)  # missing seg at t=1 -> zeros
    assert np.all(stack[0, 2] == 255)  # dmd pattern present at t=0
    assert np.all(stack[1, 2] == 0)  # no dmd pattern at t=1
