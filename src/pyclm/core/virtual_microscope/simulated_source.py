from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tiff


class TimeSeriesImageSource:
    """
    Provides frames to SimulatedMicroscopeCore one at a time.

    Frames are served in order and, when ``loop=True``, wrap around to the
    beginning so the source never runs out.
    """

    def __init__(self, frames: list[np.ndarray], loop: bool = True):
        if not frames:
            raise ValueError("TimeSeriesImageSource requires at least one frame.")
        self._frames = frames
        self._loop = loop
        self._idx = 0

    # ------------------------------------------------------------------ #
    # Class-method constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_tiff_stack(cls, path: Path, loop: bool = True) -> TimeSeriesImageSource:
        """
        Load every slice of a TIFF stack as a separate frame.

        A 2-D array is treated as a single frame.
        A 3-D array (T, H, W) is split into T frames.
        Higher-dimensional arrays are also split along axis 0.
        """
        data = tiff.imread(str(path))
        if data.ndim == 2:
            frames = [data]
        else:
            frames = [data[i] for i in range(data.shape[0])]
        return cls(frames, loop)

    @classmethod
    def from_folder(
        cls, folder: Path, pattern: str = "*.tif", loop: bool = True
    ) -> TimeSeriesImageSource:
        """
        Collect all TIFF files matching *pattern* in *folder*, sorted by name,
        and concatenate their slices into a single frame list.
        """
        folder = Path(folder)
        files = sorted(folder.glob(pattern))
        if not files:
            raise ValueError(f"No files matching '{pattern}' found in {folder}")
        frames: list[np.ndarray] = []
        for fp in files:
            data = tiff.imread(str(fp))
            if data.ndim == 2:
                frames.append(data)
            else:
                frames.extend(data[i] for i in range(data.shape[0]))
        return cls(frames, loop)

    # ------------------------------------------------------------------ #
    # Runtime interface (used by SimulatedMicroscopeCore)
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, ...]:
        return self._frames[0].shape

    def next_frame(self) -> np.ndarray:
        frame = self._frames[self._idx]
        self._idx += 1
        if self._idx >= len(self._frames):
            if self._loop:
                self._idx = 0
            else:
                self._idx = len(self._frames) - 1
        return frame
