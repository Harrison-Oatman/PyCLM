from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

import numpy as np
from h5py import File
from natsort import natsorted

from ..datatypes import AcquisitionData, SegmentationData
from ..experiments import Experiment
from ..kinds import DEFAULT_SEGMENTATION, base_kind, seg_kind
from ..measure import Regions
from ..settings import STIMULATION
from ..settings import config as _config_change
from ..settings import device_property as _property_change
from ..settings import exposure as _exposure_change
from ..settings import position as _position_change
from ..tracking import Tracks

logger = logging.getLogger(__name__)


class ROI(NamedTuple):
    x_offset: int
    y_offset: int
    width: int
    height: int


class CameraProperties(NamedTuple):
    roi: ROI
    pixel_size_um: float


class AcquiredImageRequest(NamedTuple):
    """What a pattern method needs of one channel each time it runs."""

    id: UUID
    needs_raw: bool
    needs_seg: bool
    needs_tracks: bool = False
    # how many past deliveries of this channel to keep for context.history()
    history: int = 1
    # named segmentations wanted besides the default one (add_requirement(seg="nuclei"))
    segmentations: tuple[str, ...] = ()

    @property
    def kinds(self) -> tuple[str, ...]:
        """The routing kinds this request asks for, in delivery order."""
        kinds = []
        if self.needs_raw:
            kinds.append("raw")
        if self.needs_seg:
            kinds.append("seg")
        kinds += [seg_kind(n) for n in self.segmentations if seg_kind(n) != "seg"]
        if self.needs_tracks:
            kinds.append("tracks")
        return tuple(kinds)


def _seg_names(seg) -> tuple[str, ...]:
    """Normalise the ``seg`` argument of add_requirement to segmentation names."""
    if seg is True:
        return (DEFAULT_SEGMENTATION,)
    if not seg:
        return ()
    if isinstance(seg, str):
        return (seg,)
    return tuple(str(s) for s in seg)


class DataDock:
    def __init__(self, time_seconds, requirements: list[AcquiredImageRequest]):
        self.time_seconds = time_seconds
        self.requirements = requirements
        self.data = defaultdict(dict)

        for req in requirements:
            for kind in req.kinds:
                self.data[req.id][kind] = None

        self.complete = self.check_complete()

    def _add(self, channel_id, kind: str, data):
        # ensure channel id was expected
        assert channel_id in self.data, "unexpected data passed to pattern module"

        # ensure this kind was expected and not already passed
        assert kind in self.data[channel_id], (
            f"{kind} data being passed, but not expected"
        )
        assert self.data[channel_id][kind] is None, (
            f"expected none, found {self.data[channel_id][kind]}"
        )

        self.data[channel_id][kind] = data

    def add(self, data: AcquisitionData):
        """Slot a delivery by its ``kind`` (raw, seg or tracks)."""
        self._add(data.channel_id, getattr(data, "kind", "raw"), data)

    def add_raw(self, data: AcquisitionData):
        self._add(data.channel_id, "raw", data)

    def add_seg(self, data: SegmentationData):
        self._add(data.channel_id, "seg", data)

    def add_tracks(self, data):
        self._add(data.channel_id, "tracks", data)

    def get_awaiting(self):
        awaiting = []

        for channel in self.data:
            for img in self.data[channel]:
                if self.data[channel][img] is None:
                    awaiting.append((channel, img))

        return awaiting

    def check_complete(self):
        return len(self.get_awaiting()) == 0


class ExperimentState:
    """
    Per-experiment memory in the pattern process: bounded histories of the
    deliveries a method asked for (depth = the ``history`` of its
    requirement, default 1) and of the patterns it generated.
    """

    def __init__(
        self,
        requirements: list[AcquiredImageRequest] | None = None,
        pattern_history: int = 2,
    ):
        self.depth: dict[UUID, int] = {}
        self.requested: set[tuple[UUID, str]] = set()
        for req in requirements or ():
            self.depth[req.id] = max(1, int(getattr(req, "history", 1)))
            for kind in req.kinds:
                self.requested.add((req.id, kind))
        self.history: dict[tuple[UUID, str], deque] = {}
        self.patterns: deque = deque(maxlen=max(1, int(pattern_history)))
        self.t: int | None = None
        self.time_seconds: float = 0.0
        self.generations = 0

    @classmethod
    def from_dock(cls, dock: DataDock, t: int | None = None) -> ExperimentState:
        state = cls(dock.requirements)
        state.absorb(dock, t)
        return state

    def absorb(self, dock: DataDock, t: int | None = None):
        """Append a completed dock's deliveries to the histories."""
        for channel_id, kinds in dock.data.items():
            for kind, data in kinds.items():
                if data is None:
                    continue
                key = (channel_id, kind)
                self.requested.add(key)
                if key not in self.history:
                    self.history[key] = deque(maxlen=self.depth.get(channel_id, 1))
                self.history[key].append(data)
        self.time_seconds = dock.time_seconds
        if t is not None:
            self.t = int(t)

    def record_pattern(self, pattern_id, pattern: np.ndarray):
        self.patterns.append((pattern_id, pattern))
        self.generations += 1

    def latest(self, channel_id, kind: str):
        h = self.history.get((channel_id, kind))
        return h[-1] if h else None

    def series(self, channel_id, kind: str, n: int | None = None) -> list:
        h = list(self.history.get((channel_id, kind), ()))
        return h if n is None else h[-int(n) :]


class PatternContext:
    """
    What ``PatternMethod.generate`` receives: the current deliveries for the
    channels the method asked for, their bounded history, the method's own
    previous patterns, and the timepoint. Build it on an
    :class:`ExperimentState` (the pattern process does) or on a single
    :class:`DataDock` for one-off use.
    """

    def __init__(
        self,
        source: ExperimentState | DataDock,
        experiment: Experiment,
        t: int | None = None,
        position=None,
    ):
        if isinstance(source, DataDock):
            source = ExperimentState.from_dock(source, t)
        self._state: ExperimentState = source
        self._experiment = experiment
        self._position = position
        self._requests: list = []
        self._channel_map = {
            name: ch.channel_id for name, ch in experiment.channels.items()
        }

    # ------------------------------------------------------------ timing
    @property
    def time(self) -> float:
        """Scheduled seconds since the start of the run."""
        return self._state.time_seconds

    @property
    def t(self) -> int | None:
        """Plan timepoint of the current deliveries."""
        return self._state.t

    @property
    def generation(self) -> int:
        """How many patterns this method has generated so far in the run."""
        return self._state.generations

    # ----------------------------------------------------------- helpers
    def _get_channel_id(self, channel_name: str) -> UUID:
        if channel_name not in self._channel_map:
            raise ValueError(f"Channel '{channel_name}' not found in experiment.")
        return self._channel_map[channel_name]

    def _require(self, channel_id: UUID, kind: str, what: str):
        if (channel_id, kind) not in self._state.requested:
            raise ValueError(f"{what} was not requested.")

    def _latest_array(self, channel_id: UUID, kind: str):
        data = self._state.latest(channel_id, kind)
        return None if data is None else data.data

    @staticmethod
    def _unwrap(kind: str, data):
        if kind == "tracks":
            return Tracks(data.labels, data.rows)
        return data.data

    # ----------------------------------------------------------- current
    def raw(self, channel_name: str) -> np.ndarray:
        """Get raw image for a channel."""
        cid = self._get_channel_id(channel_name)
        self._require(cid, "raw", f"Raw data for channel '{channel_name}'")
        return self._latest_array(cid, "raw")

    def segmentation(
        self, channel_name: str, name: str = DEFAULT_SEGMENTATION
    ) -> np.ndarray:
        """Label image of a channel from the ``[segmentation]`` table ``name`` (default: the default table)."""
        cid = self._get_channel_id(channel_name)
        kind = seg_kind(name)
        what = (
            f"Segmentation data for channel '{channel_name}'"
            if kind == "seg"
            else f"Segmentation '{name}' of channel '{channel_name}'"
        )
        self._require(cid, kind, what)
        return self._latest_array(cid, kind)

    def regions(self, channel_name: str, name: str = DEFAULT_SEGMENTATION) -> Regions:
        """The segmentation of a channel as :class:`~pyclm.core.measure.Regions` (measure, paint, ...)."""
        return Regions(self.segmentation(channel_name, name))

    def tracks(self, channel_name: str) -> Tracks | None:
        """Tracked objects of a channel: relabelled mask plus a per-object table."""
        cid = self._get_channel_id(channel_name)
        self._require(cid, "tracks", f"Tracks for channel '{channel_name}'")
        data = self._state.latest(cid, "tracks")
        return None if data is None else self._unwrap("tracks", data)

    def stim_raw(self):
        cid = self._experiment.stimulation.channel_id
        self._require(
            cid,
            "raw",
            "Raw data for the stimulation output was not requested. "
            "Use PatternMethod.request_stim(raw=True); it",
        )
        return self._latest_array(cid, "raw")

    def stim_seg(self):
        cid = self._experiment.stimulation.channel_id
        self._require(
            cid,
            "seg",
            "Segmentation data for the stimulation output was not requested. "
            "Use PatternMethod.request_stim(seg=True); it",
        )
        return self._latest_array(cid, "seg")

    # ----------------------------------------------------------- history
    def history(
        self,
        channel_name: str,
        kind: str = "seg",
        n: int | None = None,
        name: str | None = None,
    ) -> list:
        """
        Past deliveries of a channel, oldest first, the current one last. At
        most the ``history`` depth declared in ``add_requirement``. ``kind``
        is ``"raw"``, ``"seg"`` or ``"tracks"``; ``name`` picks a named
        segmentation when ``kind`` is ``"seg"``.
        """
        cid = self._get_channel_id(channel_name)
        if name is not None and base_kind(kind) == "seg":
            kind = seg_kind(name)
        self._require(cid, kind, f"{kind} data for channel '{channel_name}'")
        return [self._unwrap(kind, d) for d in self._state.series(cid, kind, n)]

    def stim_history(self, kind: str = "raw", n: int | None = None) -> list:
        """Past deliveries of the stimulation frame (see ``request_stim``)."""
        cid = self._experiment.stimulation.channel_id
        self._require(cid, kind, f"{kind} data for the stimulation output")
        return [self._unwrap(kind, d) for d in self._state.series(cid, kind, n)]

    def last_pattern(self) -> np.ndarray | None:
        """The pattern this method generated last time, or None on the first call."""
        return self._state.patterns[-1][1] if self._state.patterns else None

    def pattern_history(self, n: int | None = None) -> list[np.ndarray]:
        """Previous patterns, oldest first (at most ``PatternMethod.pattern_history``)."""
        patterns = [p for _, p in self._state.patterns]
        return patterns if n is None else patterns[-int(n) :]

    # ---------------------------------------------------------- settings
    # A method may change its own experiment's acquisition settings: the
    # values apply from the next timepoint (the same delay as the pattern),
    # are recorded in events.parquet, and appear as frames-table columns.
    @property
    def requests(self) -> list:
        """Setting changes requested so far in this ``generate`` call."""
        return list(self._requests)

    def _channel_config(self, channel_name: str):
        if channel_name == STIMULATION:
            return self._experiment.stimulation
        cfg = self._experiment.channels.get(channel_name)
        if cfg is None:
            raise ValueError(
                f"Channel '{channel_name}' not found in experiment "
                f"(use '{STIMULATION}' for the stimulation channel)."
            )
        return cfg

    def settings(self, channel_name: str) -> dict:
        """
        The acquisition settings of one channel of this experiment as they
        stand now: ``exposure_ms``, ``binning``, ``config_groups`` (group →
        preset) and ``device_properties`` (``"<device>-<property>"`` → value).
        ``"stimulation"`` names the stimulation channel.
        """
        cfg = self._channel_config(channel_name)
        return {
            "exposure_ms": float(cfg.exposure),
            "binning": int(cfg.binning),
            "config_groups": {g.group: g.config for g in cfg.get_config_groups()},
            "device_properties": {
                f"{d.device}-{d.property}": d.value for d in cfg.get_device_properties()
            },
        }

    def grid(self):
        """
        The :class:`~pyclm.core.grid.GridGeometry` of a grid experiment (rows,
        columns, tile shape, pitch, the tiles' order), or None for a plain
        position. ``raw()`` and the pattern are in the stitched frame either way.
        """
        return getattr(self._position, "geometry", None)

    def position(self) -> dict | None:
        """The stage position of this experiment (``x``, ``y``, ``z`` and extras), if known."""
        return None if self._position is None else dict(self._position.as_dict())

    def set_exposure(self, channel_name: str, ms: float) -> None:
        """Change a channel's exposure (ms) from the next timepoint on."""
        self._channel_config(channel_name)
        self._requests.append(_exposure_change(channel_name, ms))

    def set_config(self, channel_name: str, group: str, preset: str) -> None:
        """Select a MicroManager preset of a config group for a channel from the next timepoint on."""
        self._channel_config(channel_name)
        self._requests.append(_config_change(channel_name, group, preset))

    def set_property(self, channel_name: str, device: str, prop: str, value) -> None:
        """Set a device property (e.g. a laser intensity) for a channel from the next timepoint on."""
        self._channel_config(channel_name)
        self._requests.append(_property_change(channel_name, device, prop, value))

    def set_position(self, x=None, y=None, z=None, pfs_offset=None) -> None:
        """Move this experiment's position (absolute stage coordinates) from the next timepoint on."""
        for key, value in (("x", x), ("y", y), ("z", z), ("pfs_offset", pfs_offset)):
            if value is not None:
                self._requests.append(_position_change(key, value))


class PatternMethod:
    name = "base"
    # how many of this method's previous patterns the context keeps
    pattern_history = 2

    def __init__(
        self, experiment_name=None, camera_properties: CameraProperties = None, **kwargs
    ):
        # Support legacy init where these are passed
        self.experiment_name = experiment_name
        self.camera_properties = camera_properties
        if camera_properties:
            self.pixel_size_um = camera_properties.pixel_size_um
            self.pattern_shape = (
                camera_properties.roi.height,
                camera_properties.roi.width,
            )
        else:
            self.pixel_size_um = 1.0
            self.pattern_shape = (100, 100)  # Default placeholders

        self.binning = 1
        # (channel_name, raw, seg, tracks, history)
        self._requirements_list = []
        self._experiment_ref = None

        self._stim_requested = False
        self._stim_request_raw = False
        self._stim_request_seg = False
        self._stim_request_history = 1

    def add_requirement(
        self,
        channel_name: str,
        raw: bool = False,
        seg=False,
        tracks: bool = False,
        history: int = 1,
    ):
        """
        Declare what ``generate`` needs of a channel: the raw frame, its
        segmentation, its tracks, and how many past deliveries to keep for
        ``context.history()`` (deliveries happen at the pattern's cadence).

        ``seg`` is ``True`` for the experiment's default ``[segmentation]``
        table, the name of a ``[segmentation.<name>]`` table, or a list of
        names (``"segmentation"`` stands for the default one).
        """
        self._requirements_list.append(
            (
                channel_name,
                bool(raw),
                _seg_names(seg),
                bool(tracks),
                max(1, int(history)),
            )
        )

    def request_stim(self, raw: bool = False, seg: bool = False, history: int = 1):
        """Request the imaged stimulation"""
        self._stim_requested = True
        self._stim_request_raw = raw
        self._stim_request_seg = seg
        self._stim_request_history = max(1, int(history))

    # initialize happens shortly after init
    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        # If user used add_requirement, process them
        reqs = []
        for (
            name,
            needs_raw,
            seg_names,
            needs_tracks,
            history,
        ) in self._requirements_list:
            ch = experiment.channels.get(name)
            if ch:
                named = tuple(n for n in seg_names if n != DEFAULT_SEGMENTATION)
                for seg in named:
                    if seg not in experiment.segmentations:
                        logger.warning(
                            f"Pattern {self.name} requested segmentation {seg!r} of "
                            f"channel {name}, which the experiment does not configure"
                        )
                reqs.append(
                    AcquiredImageRequest(
                        ch.channel_id,
                        needs_raw,
                        DEFAULT_SEGMENTATION in seg_names,
                        needs_tracks,
                        history,
                        named,
                    )
                )
            else:
                logger.warning(f"Pattern {self.name} requested unknown channel {name}")

        if self._stim_requested:
            reqs.append(
                AcquiredImageRequest(
                    experiment.stimulation.channel_id,
                    self._stim_request_raw,
                    self._stim_request_seg,
                    False,
                    self._stim_request_history,
                )
            )

        return reqs

    # configure system is run after all pattern methods are initialized
    def configure_system(
        self,
        experiment_name: str,
        camera_properties: CameraProperties,
        experiment: Experiment,
    ):
        """Called by the system to inject dependencies."""
        self.experiment_name = experiment_name
        self.camera_properties = camera_properties
        self.pixel_size_um = camera_properties.pixel_size_um
        self.pattern_shape = (camera_properties.roi.height, camera_properties.roi.width)
        self._experiment_ref = experiment

        # this binning should actually take effect
        binning = experiment.stimulation.binning
        self.update_binning(binning)

    def update(self, **parameters) -> tuple[dict, dict]:
        """
        Change parameters while the run is in progress (a ``set_pattern``
        command). The default sets attributes that already exist on the
        method and refuses the rest; override it for anything smarter.
        Returns ``(applied, refused)`` where ``refused`` maps a name to the
        reason.
        """
        applied, refused = {}, {}
        for name, value in parameters.items():
            if (
                name.startswith("_")
                or not hasattr(self, name)
                or callable(getattr(self, name))
            ):
                refused[name] = "unknown parameter"
                continue
            setattr(self, name, value)
            applied[name] = value
        return applied, refused

    def get_um_meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.pattern_shape

        y_range = np.arange(h) * self.pixel_size_um
        x_range = np.arange(w) * self.pixel_size_um

        xx, yy = np.meshgrid(x_range, y_range)

        return xx, yy

    def center_um(self) -> tuple[float, float]:
        h, w = self.pattern_shape
        return (w * self.pixel_size_um / 2.0, h * self.pixel_size_um / 2.0)

    def generate(self, data_dock: DataDock | PatternContext) -> np.ndarray:
        # If passed PatternContext, user is using new API.
        # But if user implemented old generate(self, data_dock: DataDock), we need to support that.
        # This method is called by the system.
        raise NotImplementedError

    def update_binning(self, binning: int):
        binning = int(binning)
        binning_rescale = binning / self.binning

        self.pixel_size_um = self.pixel_size_um * binning_rescale

        # recover the unbinned shape, then rebin, keeping the shape integral
        h_unbinned = round(self.pattern_shape[0] * self.binning)
        w_unbinned = round(self.pattern_shape[1] * self.binning)
        self.pattern_shape = (h_unbinned // binning, w_unbinned // binning)

        logger.info(
            f"model {self.name} updated pixel size (um) to {self.pixel_size_um}"
        )
        logger.info(f"model {self.name} updated pattern_shape to {self.pattern_shape}")

        self.binning = binning
