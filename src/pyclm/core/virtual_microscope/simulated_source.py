from pathlib import Path

import numpy as np
import tifffile as tiff


class TimeSeriesImageSource:
    def __init__(self, frames: list[np.ndarray], loop: bool = True):
        if not frames:
            raise ValueError("TimeSeriesImageSource requires at least one frame.")
        self._frames = [self._normalize_frame(f) for f in frames]
        self._loop = loop
        self._idx = 0

    @classmethod
    def from_tiff_stack(cls, path: Path, loop: bool = True) -> "TimeSeriesImageSource":
        data = tiff.imread(str(path))
        if data.ndim == 2:
            frames = [data]
        elif data.ndim == 3:
            if data.shape[0] <= 4:
                frames = [data[i] for i in range(data.shape[0])]
            else:
                frames = [data]

        else:
            raise ValueError(f"Unsupported TIFF shape: {data.shape}")
        return cls(frames, loop=loop)

    @classmethod
    def from_folder(
        cls, folder: Path, pattern: str = "*.tif", loop: bool = True
    ) -> "TimeSeriesImageSource":
        paths = sorted(folder.glob(pattern))
        frames = []
        for p in paths:
            data = tiff.imread(str(p))
            if data.ndim == 2:
                frames.append(data)
            elif data.ndim == 3 and data.shape[0] <= 4:
                frames.append(data[0])
            else:
                frames.append(data)
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

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            if frame.shape[0] <= 4:
                frame = frame[0]
        return frame
