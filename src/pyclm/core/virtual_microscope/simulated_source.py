from pathlib import Path
from typing import List, Optional

import numpy as np
import tifffile as tiff


class TimeSeriesImageSource:
    def __init__(self, frames: List[np.ndarray], loop: bool = True):
        if not frames:
            raise ValueError("TimeSeriesImageSource requires at least one frame.")
        self._frames = frames
        self._loop = loop
        self._idx = 0

    @classmethod
    def from_tiff_stack(cls, path: Path, loop: bool = True) -> "TimeSeriesImageSource":
        data = tiff.imread(str(path))
        if data.ndim == 2:
            frames = [data]
        else:
            frames = [data[i] for i in range(data.shape[0])]
        return cls(frames, loop=loop)

    @classmethod
    def from_folder(cls, folder: Path, pattern: str = "*.tif", loop: bool = True) -> "TimeSeriesImageSource":
        paths = sorted(folder.glob(pattern))
        frames = [tiff.imread(str(p)) for p in paths]
        return cls(frames, loop=loop)

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