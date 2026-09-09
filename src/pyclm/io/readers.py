"""
Read PyCLM output in any format it has written: the original HDF5 layout
(format 1) and OME-Zarr (format 2). One API for the GUI, the exporter, and
analysis code.

    exp = pyclm.io.open("bar10.pos1.zarr")     # or "bar10.pos1.hdf5"
    g = exp.groups["imaging"]                   # a cadence group
    g.frame(i, "545")                           # numpy array or None if not (yet) acquired
    g.labels(i, "545")                          # segmentation labels or None
    g.labels(i, "545", "nuclei")                # labels of a named [segmentation.nuclei] table
    g.global_t(i)                               # plan timepoint of slot i
    exp.pattern_at(t)                           # DMD pattern in force at plan timepoint t
    exp.frames                                  # pyarrow.Table (format 2) or None
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class GroupData(ABC):
    """One image: channels sharing a cadence, compact time axis."""

    name: str
    channels: tuple[str, ...]
    every_t: int
    t_delay: int
    t_stop: int
    timepoints: int
    binning: int
    interval_s: float | None = None  # plan interval in seconds
    pixel_size_um: float | None = None  # at this group's binning

    def local_index(self, t: int) -> int | None:
        this_t = t - self.t_delay
        if this_t < 0 or this_t % self.every_t != 0:
            return None
        if self.t_stop > 0 and this_t >= self.t_stop:
            return None
        return this_t // self.every_t

    def global_t(self, i: int) -> int:
        return self.t_delay + i * self.every_t

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """(height, width) of one frame."""

    @abstractmethod
    def acquired(self) -> list[int]:
        """Slots that hold at least one written channel, ascending."""

    @abstractmethod
    def frame(self, i: int, channel: str) -> np.ndarray | None: ...

    @abstractmethod
    def labels(
        self, i: int, channel: str, name: str = "segmentation"
    ) -> np.ndarray | None:
        """Label image of a slot from the segmentation table ``name``, or None."""

    @property
    def has_labels(self) -> bool:
        return bool(self.label_names)

    @property
    def label_names(self) -> tuple[str, ...]:
        """Segmentation tables stored for this group (the default one is ``"segmentation"``)."""
        return ()

    def tracks(self, i: int, channel: str) -> np.ndarray | None:
        """Label image relabelled with track ids for a slot, or None."""
        return None

    @property
    def has_tracks(self) -> bool:
        return False


class ExperimentData(ABC):
    """Everything PyCLM stored for one experiment (one position)."""

    format: int
    path: Path
    name: str
    groups: dict[str, GroupData]
    affine_transform: np.ndarray | None
    slm_shape: tuple[int, int] | None
    plan_yaml: str | None

    @property
    @abstractmethod
    def current_t(self) -> int:
        """Highest plan timepoint whose saved frames are all written, or -1."""

    @abstractmethod
    def pattern_at(self, t: int) -> np.ndarray | None:
        """DMD-space pattern in force at plan timepoint ``t`` (latest at or before ``t``)."""

    @property
    def frames(self):
        """The frames table rows for this experiment (pyarrow), or None."""
        return None

    @property
    def tracks(self):
        """The tracks table rows for this experiment (pyarrow), or None."""
        return None

    @property
    def events(self):
        """
        Runtime events of this experiment (pyarrow): setting changes a
        pattern method requested, late timepoints, acquisition errors. From
        ``events.parquet`` in the experiment directory; None if absent.
        """
        path = Path(self.path).parent / "events.parquet"
        if not path.exists():
            return None
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        mask = np.asarray(
            table["experiment"].to_numpy(zero_copy_only=False) == self.name
        )
        return table.filter(mask)

    def refresh(self) -> None:
        """Re-read metadata that changes while the experiment is running."""
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================ OME-Zarr (format 2)
class _ZarrGroup(GroupData):
    @staticmethod
    def _path_for(root_path, name):
        return root_path / name

    def __init__(
        self,
        root_path: Path,
        name: str,
        meta: dict,
        label_names=(),
        has_tracks: bool = False,
    ):
        import zarr

        self.name = name
        self.channels = tuple(meta["channels"])
        self.every_t = int(meta["every_t"])
        self.t_delay = int(meta["t_delay"])
        self.t_stop = int(meta["t_stop"])
        self.timepoints = int(meta["timepoints"])
        self.binning = int(meta["binning"])
        self.interval_s = meta.get("interval_seconds")
        self.pixel_size_um = None
        try:
            import zarr

            scale = zarr.open_group(
                str(self._path_for(root_path, name)), mode="r"
            ).attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
                "scale"
            ]
            self.pixel_size_um = float(scale[-1])
            if self.interval_s is None and self.every_t:
                self.interval_s = float(scale[0]) / self.every_t
        except Exception:
            pass
        self._path = root_path / name
        self._arr = zarr.open_array(str(self._path / "0"), mode="r")
        self._labels = {
            n: zarr.open_array(str(self._path / "labels" / n / "0"), mode="r")
            for n in label_names
        }
        self._tracks = (
            zarr.open_array(str(self._path / "labels" / "tracks" / "0"), mode="r")
            if has_tracks
            else None
        )

    @property
    def shape(self):
        return tuple(int(v) for v in self._arr.shape[2:])

    @property
    def label_names(self):
        return tuple(self._labels)

    @property
    def has_tracks(self):
        return self._tracks is not None

    def _chunk_exists(self, i: int, c: int, labels=None, tracks=False) -> bool:
        if tracks:
            base = self._path / "labels" / "tracks" / "0"
        elif labels:
            base = self._path / "labels" / str(labels) / "0"
        else:
            base = self._path / "0"
        return (base / f"{i}.{c}.0.0").exists()

    def acquired(self):
        return [
            i
            for i in range(self._arr.shape[0])
            if any(self._chunk_exists(i, c) for c in range(len(self.channels)))
        ]

    def frame(self, i, channel):
        c = self.channels.index(channel)
        if not self._chunk_exists(i, c):
            return None
        return np.asarray(self._arr[i, c])

    def labels(self, i, channel, name="segmentation"):
        arr = self._labels.get(name)
        if arr is None:
            return None
        c = self.channels.index(channel)
        if not self._chunk_exists(i, c, labels=name):
            return None
        return np.asarray(arr[i, c])

    def tracks(self, i, channel):
        if self._tracks is None:
            return None
        c = self.channels.index(channel)
        if not self._chunk_exists(i, c, tracks=True):
            return None
        return np.asarray(self._tracks[i, c])


class ZarrExperiment(ExperimentData):
    format = 2

    def __init__(self, path):
        import zarr

        self.path = Path(path)
        self._root = zarr.open_group(str(self.path), mode="r")
        meta = dict(self._root.attrs["pyclm"])
        self._meta = meta
        self.name = meta["experiment"]
        self.plan_yaml = meta.get("plan")
        at = meta.get("affine_transform")
        self.affine_transform = None if at is None else np.asarray(at, dtype=np.float32)
        self.slm_shape = (
            None if meta.get("slm_shape") is None else tuple(meta["slm_shape"])
        )
        self.groups = {
            name: _ZarrGroup(
                self.path,
                name,
                gmeta,
                self._label_names(self.path / name / "labels"),
                (self.path / name / "labels" / "tracks" / "0").exists(),
            )
            for name, gmeta in meta["groups"].items()
        }
        self._frames_path = self.path.parent / "frames.parquet"
        self._frames = None
        self._tracks_path = self.path.parent / "tracks.parquet"
        self._tracks = None
        self._pattern_arr = None
        if (self.path / "patterns" / "dmd" / "0").exists():
            self._pattern_arr = zarr.open_array(
                str(self.path / "patterns" / "dmd" / "0"), mode="r"
            )

    @staticmethod
    def _label_names(labels_dir: Path) -> tuple[str, ...]:
        """Segmentation label images under a group (NGFF ``labels`` list, minus tracks)."""
        attrs = labels_dir / ".zattrs"
        names: list[str] = []
        if attrs.exists():
            try:
                names = list(json.loads(attrs.read_text()).get("labels", []))
            except Exception:
                names = []
        if not names and labels_dir.exists():
            names = sorted(p.name for p in labels_dir.iterdir() if p.is_dir())
        return tuple(
            n for n in names if n != "tracks" and (labels_dir / n / "0").exists()
        )

    def refresh(self):
        import zarr

        self._root = zarr.open_group(str(self.path), mode="r")
        self._meta = dict(self._root.attrs["pyclm"])
        self._frames = None
        self._tracks = None
        for g in self.groups.values():
            g._arr = zarr.open_array(str(g._path / "0"), mode="r")
            for n in list(g._labels):
                g._labels[n] = zarr.open_array(
                    str(g._path / "labels" / n / "0"), mode="r"
                )
            if g._tracks is not None:
                g._tracks = zarr.open_array(
                    str(g._path / "labels" / "tracks" / "0"), mode="r"
                )
        if self._pattern_arr is not None:
            self._pattern_arr = zarr.open_array(
                str(self.path / "patterns" / "dmd" / "0"), mode="r"
            )

    @property
    def current_t(self):
        return int(self._meta.get("current_t", -1))

    def _read_rows(self, path):
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        mask = np.asarray(
            table["experiment"].to_numpy(zero_copy_only=False) == self.name
        )
        return table.filter(mask)

    @property
    def frames(self):
        if self._frames is None and self._frames_path.exists():
            self._frames = self._read_rows(self._frames_path)
        return self._frames

    @property
    def tracks(self):
        if self._tracks is None and self._tracks_path.exists():
            self._tracks = self._read_rows(self._tracks_path)
        return self._tracks

    @property
    def routing(self) -> dict | None:
        """The resolved routing table recorded at the start of the run, if any."""
        return self._meta.get("routing")

    def _pattern_index_at(self, t: int) -> int | None:
        table = self.frames
        if table is None or table.num_rows == 0:
            return None
        kinds = table["kind"].to_pylist()
        ts = table["t"].to_pylist()
        idx = table["pattern_index"].to_pylist()
        best = None
        for k, tt, pi in zip(kinds, ts, idx, strict=False):
            if (
                k == "stim_event"
                and pi is not None
                and tt <= t
                and (best is None or tt >= best[0])
            ):
                best = (tt, pi)
        return None if best is None else int(best[1])

    def pattern_at(self, t):
        if self._pattern_arr is None:
            return None
        i = self._pattern_index_at(t)
        if i is None or i >= self._pattern_arr.shape[0]:
            return None
        return np.asarray(self._pattern_arr[i])

    @property
    def pattern_ids(self) -> list[str]:
        try:
            return list(self._root["patterns/dmd"].attrs["pyclm"]["pattern_ids"])
        except KeyError:
            return []


# ============================================================ HDF5 (format 1)
class _HDF5Group(GroupData):
    """One channel of the v1 layout, presented as a single-channel group."""

    def __init__(self, exp: HDF5Experiment, key: str, every_t: int):
        self._exp = exp
        self.key = key  # "channel_545" or "stim_aq"
        self.name = key if key == "stim_aq" else key.replace("channel_", "imaging_", 1)
        self.channels = (
            "stim" if key == "stim_aq" else key.replace("channel_", "", 1),
        )
        self.every_t = every_t
        self.t_delay = exp._t_delay
        self.t_stop = exp._t_stop
        self.timepoints = sum(
            1 for t in range(exp._t_count) if self.local_index(t) is not None
        )
        self.binning = exp._binning_for(key)
        self.interval_s = exp._interval_s
        self.pixel_size_um = None
        self._shape = None

    @property
    def shape(self):
        if self._shape is None:
            self._read_pixel_size()
            for i in self.acquired():
                fr = self.frame(i, self.channels[0])
                if fr is not None:
                    self._shape = fr.shape
                    break
        return self._shape or (0, 0)

    @property
    def label_names(self):
        # v1 pre-allocates empty seg datasets for every channel; only a written one counts
        if any(self._dset(i, "seg") is not None for i in self.acquired()):
            return ("segmentation",)
        return ()

    def _read_pixel_size(self):
        f = self._exp._file
        for i in self.acquired():
            key = f"{self.global_t(i):05d}/{self.key}/data"
            val = f[key].attrs.get("pixel_width_um", "")
            try:
                self.pixel_size_um = float(val)
                return
            except (TypeError, ValueError):
                continue

    def _dset(self, i, name):
        f = self._exp._file
        key = f"{self.global_t(i):05d}/{self.key}/{name}"
        if key not in f:
            return None
        d = f[key]
        try:
            d.id.refresh()
        except Exception:
            pass
        if 0 in d.shape:
            return None
        return np.asarray(d)

    def acquired(self):
        return [i for i in range(self.timepoints) if self._dset(i, "data") is not None]

    def frame(self, i, channel):
        return self._dset(i, "data")

    def labels(self, i, channel, name="segmentation"):
        if name != "segmentation":
            return None
        return self._dset(i, "seg")


class HDF5Experiment(ExperimentData):
    format = 1

    def __init__(self, path):
        import h5py

        self.path = Path(path)
        self.name = self.path.stem
        self._file = h5py.File(str(self.path), mode="r", libver="latest", swmr=True)
        attrs = self._file.attrs
        self.plan_yaml = attrs.get("plan")
        at = attrs.get("affine_transform")
        self.affine_transform = None if at is None else np.asarray(at, dtype=np.float32)
        self.slm_shape = None
        self._t_delay = int(attrs.get("t_delay", 0))
        self._t_stop = int(attrs.get("t_stop", 0))
        self._t_count = int(attrs.get("t_count", 0)) or len(
            [k for k in self._file.keys() if k[:1].isdigit()]
        )
        every_t = json.loads(attrs.get("every_t", "{}"))
        self._experiment_meta = json.loads(attrs.get("experiment_metadata", "{}"))
        schedule_meta = json.loads(attrs.get("schedule_metadata", "{}"))
        self._interval_s = schedule_meta.get("times", {}).get("interval")
        self.groups = {}
        for key, n in every_t.items():
            g = _HDF5Group(self, key, int(n))
            self.groups[g.name] = g

    def _binning_for(self, key: str) -> int:
        meta = self._experiment_meta
        if key == "stim_aq":
            return int(meta.get("stimulation", {}).get("binning", 1))
        short = key.replace("channel_", "", 1)
        return int(meta.get("channels", {}).get(short, {}).get("binning", 1))

    def refresh(self):
        # SWMR readers on Windows do not always see new data through one handle
        if sys.platform == "win32":
            import h5py

            self._file.close()
            self._file = h5py.File(str(self.path), mode="r", libver="latest", swmr=True)

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    @property
    def current_t(self):
        try:
            d = self._file["current_t_index"]
            d.id.refresh()
            return int(d[()])
        except Exception:
            return -1

    def pattern_at(self, t):
        f = self._file
        for tt in range(t, -1, -1):
            key = f"{tt:05d}/stim_aq/dmd"
            if key in f:
                d = f[key]
                try:
                    d.id.refresh()
                except Exception:
                    pass
                if 0 not in d.shape:
                    return np.asarray(d)
        return None


def open(path) -> ExperimentData:
    """Open a PyCLM experiment output, whichever format it was written in."""
    path = Path(path)
    if path.is_dir() and (path / ".zattrs").exists():
        return ZarrExperiment(path)
    if path.suffix in (".hdf5", ".h5"):
        return HDF5Experiment(path)
    raise ValueError(
        f"{path} is not a PyCLM experiment output (.zarr directory or .hdf5 file)"
    )


def find_experiments(directory) -> list[Path]:
    """All experiment outputs in a directory, newest layout first."""
    directory = Path(directory)
    zarrs = sorted(p for p in directory.glob("*.zarr") if (p / ".zattrs").exists())
    h5s = sorted(directory.glob("*.hdf5"))
    return zarrs + h5s
