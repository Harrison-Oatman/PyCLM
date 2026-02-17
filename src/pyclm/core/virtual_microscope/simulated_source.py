from pathlib import Path

import numpy as np
import tifffile as tiff
import yaml


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

    def read_yaml(self, data_directory: Path):
        yaml_path = Path.joinpath(data_directory, "image_positions.yaml")
        with open(yaml_path) as file:
            data = yaml.safe_load(file)
        if data is None:
            raise ValueError(f"{yaml_path} is empty")
        positions = data.get("positions")
        pos_map: dict[tuple[float, float], str] = {}
        for entry in positions:
            x = entry["x"]
            y = entry["y"]
            file = str(entry["file"])
            key = (x, y)
            pos_map[key] = file
        return pos_map

    def initialize_from_folder(self, folder: Path) -> None:
        folder = Path(folder)
        self._pos_map = self.read_yaml(folder)
        self._index_map: dict[str, int] = {}
        self._frames_map: dict[str, list[np.ndarray]] = {}
        for name in self._pos_map.values():
            file_path = Path.joinpath(folder, Path(name))
            data = self._load_and_normalize_tiff(file_path)
            frames = [data[t, 0, 0] for t in range(data.shape[0])]
            self._frames_map[name] = frames
            self._index_map[name] = 0
        self._default_stack = next(iter(self._frames_map.keys()))

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

    @staticmethod
    def _normalize_axes_ome(data: np.ndarray, axes: str) -> np.ndarray:
        target = "TCZYX"
        axes = axes.upper()
        axis_map = {ax: i for i, ax in enumerate(axes)}
        for ax in target:
            if ax not in axis_map:
                data = np.expand_dims(data, axis=0)
                axis_map = {k: v + 1 for k, v in axis_map.items()}
                axis_map[ax] = 0

        order = [axis_map[ax] for ax in target]
        return np.moveaxis(data, order, range(len(order)))

    def _load_and_normalize_tiff(self, file_path: Path) -> np.ndarray:
        with tiff.TiffFile(file_path) as tif:
            data = tif.asarray()
            if tif.is_ome:
                axes = tif.series[0].axes
                data = self._normalize_axes_ome(data, axes)
            else:
                raise ValueError(
                    f"{file_path} is not an OME-TIFF. This loader requires OME axis metadata."
                )
        if data.ndim != 5:
            raise RuntimeError("TIFF data not 5D after normalization")
        return data
