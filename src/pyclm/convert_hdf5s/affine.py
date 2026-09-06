from pathlib import Path

import cv2
import numpy as np
from toml import load

# (x_offset, y_offset, width, height), as returned by core.getROI()
ROI = tuple[int, int, int, int]


class AffineCalibration:
    """
    Camera -> SLM affine transform (2x3 matrix), calibrated against the full,
    unbinned camera sensor (ROI offset (0, 0)).

    Provides `warp_pattern_to_camera`, which takes a DMD pattern (in SLM
    coordinates) and warps it directly into the binned, ROI-cropped camera
    frame used for a given acquired channel - in a single `cv2.warpAffine`
    call sized to the *output* shape, with no intermediate upscaled array.

    A small cache of recently-seen patterns avoids repeating this warp when
    the DMD pattern is static across consecutive (or near-consecutive)
    frames.
    """

    def __init__(self, transform: np.ndarray, cache_size: int = 2):
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (2, 3):
            raise ValueError("Affine transform must be a 2x3 matrix")

        self.transform = transform
        self.cache_size = cache_size
        self._cache: list[tuple] = []

    @classmethod
    def from_config(
        cls, config_path: Path | str, cache_size: int = 2
    ) -> "AffineCalibration":
        config = load(Path(config_path))
        transform = np.array(config["affine_transform"], dtype=np.float64)
        return cls(transform, cache_size=cache_size)

    def warp_pattern_to_camera(
        self,
        pattern: np.ndarray,
        out_shape: tuple[int, int],
        binning: int = 1,
        roi: ROI | None = None,
    ) -> np.ndarray:
        """
        Warp `pattern` (uint8 array, SLM coordinates) into the camera frame
        described by `out_shape` (binned, ROI-cropped (H, W)), returning a
        uint16 array of shape `out_shape`.
        """
        roi_offset = (roi[0], roi[1]) if roi is not None else (0, 0)

        for (
            cached_pattern,
            cached_binning,
            cached_roi,
            cached_shape,
            result,
        ) in self._cache:
            if (
                cached_binning == binning
                and cached_roi == roi_offset
                and cached_shape == out_shape
                and np.array_equal(cached_pattern, pattern)
            ):
                return result

        result = self._warp(pattern, out_shape, binning, roi_offset)

        if self.cache_size > 0:
            self._cache.insert(0, (pattern, binning, roi_offset, out_shape, result))
            del self._cache[self.cache_size :]

        return result

    def _warp(
        self,
        pattern: np.ndarray,
        out_shape: tuple[int, int],
        binning: int,
        roi_offset: tuple[int, int],
    ) -> np.ndarray:
        h, w = out_shape
        b = float(binning)
        x0, y0 = roi_offset

        # Output pixel (x, y) covers unbinned camera pixels
        # [x0 + x*b, x0 + (x+1)*b) x [y0 + y*b, y0 + (y+1)*b); sample at the
        # center of that block, i.e. unbinned camera coords
        # (x0 + x*b + (b-1)/2, y0 + y*b + (b-1)/2).
        at = self.transform
        center = np.array([x0 + (b - 1) / 2, y0 + (b - 1) / 2])

        m = np.empty((2, 3), dtype=np.float32)
        m[:, :2] = at[:, :2] * b
        m[:, 2] = at[:, :2] @ center + at[:, 2]

        source = np.round(pattern).astype(np.uint8)
        warped = cv2.warpAffine(
            source,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        return warped.astype(np.uint16)
