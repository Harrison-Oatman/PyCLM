from pathlib import Path

import numpy as np
import tifffile as tiff
import yaml


class TimeSeriesImageSource:
    def __init__(self, folder: Path, loop: bool = True):
        self._loop = loop
        self._pos_map: dict[tuple[float, float], str] = {}
        self._index_map: dict[str, int] = {}
        self._frames_map: dict[str, list[np.ndarray]] = {}
        self._default_stack: str | None = None
        self._use_default_for_unknown: bool = False

        self.initialize_from_folder(folder)

    @classmethod
    def from_mapping(
        cls,
        pos_to_tif: dict[tuple[float, float], Path],
        loop: bool = True,
    ) -> "TimeSeriesImageSource":
        """Build a source directly from a coordinate → TIF-path mapping."""
        instance = cls.__new__(cls)
        instance._loop = loop
        instance._pos_map = {}
        instance._index_map = {}
        instance._frames_map = {}
        instance._default_stack = None
        instance._use_default_for_unknown = False

        for pos, tif_path in pos_to_tif.items():
            tif_path = Path(tif_path)
            name = str(tif_path)
            instance._pos_map[pos] = name
            if name not in instance._frames_map:
                data = instance._load_and_normalize_tiff(tif_path)
                frames = [data[t, 0, 0] for t in range(data.shape[0])]
                instance._frames_map[name] = frames
                instance._index_map[name] = 0

        if instance._frames_map:
            instance._default_stack = next(iter(instance._frames_map.keys()))

        return instance

    @classmethod
    def from_tiff_stack(
        cls,
        tif_path: Path,
        loop: bool = True,
    ) -> "TimeSeriesImageSource":
        """Build a source from a single TIF stack used for all positions."""
        instance = cls.__new__(cls)
        instance._loop = loop
        instance._pos_map = {}
        instance._index_map = {}
        instance._frames_map = {}
        instance._default_stack = None
        instance._use_default_for_unknown = True

        tif_path = Path(tif_path)
        name = str(tif_path)
        data = instance._load_and_normalize_tiff(tif_path)
        frames = [data[t, 0, 0] for t in range(data.shape[0])]
        instance._frames_map[name] = frames
        instance._index_map[name] = 0
        instance._default_stack = name

        return instance

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
        if self._default_stack is None:
            raise RuntimeError("TimeSeriesImageSource not initialized")
        return self._frames_map[self._default_stack][0].shape

    def next_frame(self, pos) -> np.ndarray:
        pos = tuple(pos)
        if pos not in self._pos_map:
            if self._use_default_for_unknown and self._default_stack is not None:
                stack_name = self._default_stack
            else:
                raise KeyError(f"Position {pos} not found in image source")
        else:
            stack_name = self._pos_map[pos]
        frames = self._frames_map[stack_name]
        frame = frames[self._index_map[stack_name]]
        self._index_map[stack_name] += 1
        if self._index_map[stack_name] >= len(frames):
            if self._loop:
                self._index_map[stack_name] = 0
            else:
                self._index_map[stack_name] = len(frames) - 1
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
                # Non-OME fallback: infer axes from ndim, then normalize to TCZYX
                # Heuristics (common cases):
                #   2D: YX
                #   3D: TYX
                #   4D: TCYX
                #   5D: TCZYX
                if data.ndim == 2:
                    axes = "YX"
                elif data.ndim == 3:
                    axes = "TYX"
                elif data.ndim == 4:
                    axes = "TCYX"
                elif data.ndim == 5:
                    axes = "TCZYX"
                else:
                    raise ValueError(
                        f"{file_path} has unsupported ndim={data.ndim}. "
                        "Expected 2D-5D TIFF for non-OME fallback."
                    )

                data = self._normalize_axes_ome(data, axes)

        if data.ndim != 5:
            raise RuntimeError("TIFF data not 5D after normalization")
        return data
