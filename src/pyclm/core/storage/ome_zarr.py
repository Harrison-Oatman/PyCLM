"""
OME-Zarr (NGFF 0.4, zarr v2) writer: PyCLM storage format 2.

Layout per experiment (``<experiment>.zarr/``):

    .zattrs                     pyclm: {format: 2, plan, experiment, schedule, affine_transform,
                                slm_shape, groups: {...}, current_t, useq_version}
    <group>/                    one NGFF image per cadence group ("imaging", "stim", ...)
        .zattrs                 multiscales (axes t,c,y,x; time scale = every_t * interval), omero
        0                       (T, C, Y, X) uint16, chunks (1, 1, Y, X), zstd; compact T
        labels/segmentation/0   (T, C, Y, X) uint16 label images (only if segmentation.save)
        labels/<name>/0         the same for each named [segmentation.<name>] table
    patterns/dmd/0              (N, H_slm, W_slm) uint8, one per distinct pattern_id
                                (policy "on_change"), or one per stimulation event ("all")
    frames.parquet / frames.csv one row per acquired frame and per stimulation event

Skipped timepoints are never written: a chunk that was not acquired does not
exist on disk and reads as the fill value.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numcodecs
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import zarr

from ..core_interface import MicroscopeCoreInterface
from ..datatypes import AcquisitionData, SegmentationData, StimulationData
from ..kinds import DEFAULT_SEGMENTATION, is_seg_kind, seg_name
from ..plan import AcquisitionPlan
from .base import (
    PATTERN_POLICIES,
    CadenceGroup,
    FrameWriter,
    cadence_groups,
    image_shape,
)

logger = logging.getLogger(__name__)

STORAGE_FORMAT_VERSION = 3
NGFF_VERSION = "0.4"
FRAMES_COLUMNS = (
    "experiment",
    "kind",
    "t",
    "group",
    "local_index",
    "channel",
    "scheduled_at",
    "completed_at",
    "exposure_ms",
    "binning",
    "x",
    "y",
    "z",
    "pfs_offset",
    "pattern_id",
    "pattern_index",
    "dmd_index",
    "pixel_size_um",
)
TRACKS_COLUMNS = (
    "experiment",
    "t",
    "channel",
    "group",
    "local_index",
    "track_id",
    "label",
    "y",
    "x",
    "y_um",
    "x_um",
    "area",
    "parent",
)
TRACKS_SCHEMA = pa.schema(
    [
        ("experiment", pa.string()),
        ("t", pa.int32()),
        ("channel", pa.string()),
        ("group", pa.string()),
        ("local_index", pa.int32()),
        ("track_id", pa.int64()),
        ("label", pa.int64()),
        ("y", pa.float64()),
        ("x", pa.float64()),
        ("y_um", pa.float64()),
        ("x_um", pa.float64()),
        ("area", pa.int64()),
        ("parent", pa.int64()),
    ]
)


def _iso(timestamp) -> str | None:
    if not timestamp:
        return None
    return datetime.datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")


def _compressor():
    return numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)


def ngff_image_attrs(
    name, axes_scale, channel_names, unit_time="second", unit_space="micrometer"
):
    """NGFF 0.4 multiscales + omero attributes for a (t, c, y, x) image."""
    return {
        "multiscales": [
            {
                "version": NGFF_VERSION,
                "name": name,
                "axes": [
                    {"name": "t", "type": "time", "unit": unit_time},
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": unit_space},
                    {"name": "x", "type": "space", "unit": unit_space},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [float(v) for v in axes_scale]}
                        ],
                    }
                ],
            }
        ],
        "omero": {
            "channels": [
                {"label": c, "active": True, "color": "FFFFFF"} for c in channel_names
            ]
        },
    }


class _ExperimentStore:
    """Open zarr handles and bookkeeping for one experiment."""

    def __init__(self, root: zarr.Group, groups: list[CadenceGroup]):
        self.root = root
        self.groups = {g.name: g for g in groups}
        self.images: dict[str, zarr.Array] = {}
        # (cadence group, segmentation name) -> label array
        self.labels: dict[tuple[str, str], zarr.Array] = {}
        self.tracks: dict[str, zarr.Array] = {}
        self.pattern_array: zarr.Array | None = None  # patterns/dmd/0
        self.pattern_ids: list[str] = []  # one per DMD image
        self.dmd_camera_ids: list[
            str
        ] = []  # the camera pattern each DMD image came from
        self.dmd_tiles: list = []  # [row, col] of the tile, None for a plain experiment
        self.camera_array: zarr.Array | None = None  # patterns/camera/0
        self.camera_ids: list[str] = []
        self.geometry = None  # GridGeometry of a grid experiment
        self.written: set[tuple[int, str]] = set()  # (t, channel) frames written
        self.written_groups: set[str] = set()  # groups with at least one frame
        self.current_t = -1

    def group_for(self, channel: str) -> CadenceGroup | None:
        for g in self.groups.values():
            if channel in g.channels:
                return g
        return None


class OMEZarrWriter(FrameWriter):
    format = "ome-zarr"

    def __init__(self, pattern_policy: str = "on_change"):
        super().__init__()
        if pattern_policy not in PATTERN_POLICIES:
            raise ValueError(f"pattern_policy must be one of {PATTERN_POLICIES}")
        self.pattern_policy = pattern_policy
        self.stores: dict[str, _ExperimentStore] = {}
        self.rows: list[dict] = []
        self.track_rows: list[dict] = []
        self.dropped_frames = 0
        self.error_count = 0
        self.frames_path: Path | None = None
        self.tracks_path: Path | None = None
        self.routing: dict | None = None
        self._pixel_size_um: float = 1.0

    def planned_paths(self, names, base_path):
        return {name: Path(base_path) / f"{name}.zarr" for name in names}

    def output_paths(self):
        return (
            self.planned_paths(self.plan.experiments, self.base_path)
            if self.plan
            else {}
        )

    # ---------------------------------------------------------------- open
    def open(
        self,
        plan,
        core,
        base_path,
        affine_transform=None,
        slm_shape=None,
        recorded=None,
        routing=None,
    ):
        self.plan = plan
        self.recorded = recorded
        self.routing = routing
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.base_path / "frames.parquet"
        self.tracks_path = self.base_path / "tracks.parquet"
        self._pixel_size_um = float(core.getPixelSizeUm())

        slm_device = core.getSLMDevice()
        if slm_shape is None and slm_device:
            slm_shape = (core.getSLMHeight(slm_device), core.getSLMWidth(slm_device))

        all_layers = []
        try:
            for exp_name in plan.experiments:
                path = self.base_path / f"{exp_name}.zarr"
                if path.exists():
                    raise FileExistsError(f"output already exists: {path}")
                all_layers += self._open_experiment(
                    exp_name, path, core, affine_transform, slm_shape
                )
        except Exception:
            self.close()
            raise

        self._write_frames_table()
        self.is_open = True
        return all_layers

    def _open_experiment(self, exp_name, path, core, affine_transform, slm_shape):
        plan = self.plan
        experiment = plan.schedule.experiments[exp_name]
        groups = cadence_groups(plan, exp_name)
        geometry = getattr(plan.schedule.positions.get(exp_name), "geometry", None)
        root = zarr.open_group(str(path), mode="w", zarr_format=2)
        store = _ExperimentStore(root, groups)
        store.geometry = geometry
        layers = []

        for g in groups:
            h, w = image_shape(core, g.binning, geometry)
            px = self._pixel_size_um * g.binning
            img_group = root.create_group(g.name)
            img_group.attrs.update(
                ngff_image_attrs(
                    f"{exp_name}/{g.name}",
                    [g.every_t * plan.interval_s, 1.0, px, px],
                    g.channels,
                )
            )
            img_group.attrs["pyclm"] = {
                "channels": list(g.channels),
                "stim_channel": g.stim_channel,
                "every_t": g.every_t,
                "t_delay": g.t_delay,
                "t_stop": g.t_stop,
                "binning": g.binning,
                "timepoints": g.timepoints,
                "interval_seconds": plan.interval_s,
            }
            store.images[g.name] = img_group.create_array(
                "0",
                shape=(g.timepoints, len(g.channels), h, w),
                chunks=(1, 1, h, w),
                dtype="uint16",
                compressors=_compressor(),
                fill_value=0,
            )
            seg_names = self._recorded_segmentations(exp_name, g.channels)
            record_tracks = any(self.records(exp_name, c, "tracks") for c in g.channels)
            if seg_names or record_tracks:
                labels = img_group.create_group("labels")
                labels.attrs["labels"] = [
                    *seg_names,
                    *(["tracks"] if record_tracks else []),
                ]
                scale = [g.every_t * plan.interval_s, 1.0, px, px]
                for seg in seg_names:
                    store.labels[(g.name, seg)] = self._label_array(
                        labels, seg, exp_name, g, scale, (h, w), "uint16"
                    )
                if record_tracks:
                    store.tracks[g.name] = self._label_array(
                        labels, "tracks", exp_name, g, scale, (h, w), "uint32"
                    )
            for c in g.channels:
                layers.append((str(path.resolve()), f"{g.name}/{c}"))

        if slm_shape is not None and self.pattern_policy != "none":
            patterns = root.create_group("patterns/dmd")
            patterns.attrs["pyclm"] = {
                "policy": self.pattern_policy,
                "pattern_ids": [],
                "camera_ids": [],
                "tiles": [],
            }
            store.pattern_array = patterns.create_array(
                "0",
                shape=(0, int(slm_shape[0]), int(slm_shape[1])),
                chunks=(1, int(slm_shape[0]), int(slm_shape[1])),
                dtype="uint8",
                compressors=_compressor(),
                fill_value=0,
            )
            # the camera-space pattern the method returned, at the stimulation
            # binning in the ROI frame; resized to the first pattern if the
            # camera reports a different ROI (as the image groups are)
            ch, cw = image_shape(core, experiment.stimulation.binning, geometry)
            camera = root.create_group("patterns/camera")
            camera.attrs["pyclm"] = {"policy": self.pattern_policy, "pattern_ids": []}
            store.camera_array = camera.create_array(
                "0",
                shape=(0, int(ch), int(cw)),
                chunks=(1, int(ch), int(cw)),
                dtype="uint8",
                compressors=_compressor(),
                fill_value=0,
            )

        root.attrs["pyclm"] = {
            "format": STORAGE_FORMAT_VERSION,
            "ngff_version": NGFF_VERSION,
            "useq_version": _pkg_version("useq-schema"),
            "experiment": exp_name,
            "plan": plan.yaml_str(),
            "experiment_metadata": experiment.as_dict(),
            "schedule_metadata": plan.schedule.as_dict(),
            "affine_transform": None
            if affine_transform is None
            else np.asarray(affine_transform, dtype=float).tolist(),
            "slm_shape": None if slm_shape is None else [int(v) for v in slm_shape],
            "camera_roi": [int(v) for v in core.getROI()],
            "grid": None if geometry is None else geometry.as_dict(),
            "pattern_policy": self.pattern_policy,
            "routing": self.routing,
            "groups": {
                g.name: {
                    "channels": list(g.channels),
                    "every_t": g.every_t,
                    "t_delay": g.t_delay,
                    "t_stop": g.t_stop,
                    "binning": g.binning,
                    "timepoints": g.timepoints,
                }
                for g in groups
            },
            "plan_timepoints": plan.timepoints,
            "current_t": -1,
        }
        self.stores[exp_name] = store
        logger.info(f"Initialized OME-Zarr store {path}")
        return layers

    def _recorded_segmentations(self, exp_name: str, channels) -> list[str]:
        """Segmentation tables whose labels will reach the writer for these channels, default first."""
        if self.recorded is not None:
            names = {
                seg_name(kind)
                for kind, pairs in self.recorded.items()
                if is_seg_kind(kind) and any((exp_name, c) in pairs for c in channels)
            }
        else:
            exp = self.plan.schedule.experiments[exp_name]
            names = set()
            if any(self.records(exp_name, c, "seg") for c in channels):
                names.add(DEFAULT_SEGMENTATION)
            for n, cfg in exp.segmentations.items():
                if n != DEFAULT_SEGMENTATION and cfg.method_name != "none" and cfg.save:
                    names.add(n)
        return sorted(names, key=lambda n: (n != DEFAULT_SEGMENTATION, n))

    @staticmethod
    def _label_array(labels, name, exp_name, g, scale, shape, dtype):
        """One NGFF label image (segmentation or tracks) under a cadence group."""
        h, w = shape
        grp = labels.create_group(name)
        grp.attrs.update(
            ngff_image_attrs(f"{exp_name}/{g.name}/{name}", scale, g.channels)
        )
        grp.attrs["image-label"] = {
            "version": NGFF_VERSION,
            "source": {"image": "../../"},
        }
        return grp.create_array(
            "0",
            shape=(g.timepoints, len(g.channels), h, w),
            chunks=(1, 1, h, w),
            dtype=dtype,
            compressors=_compressor(),
            fill_value=0,
        )

    # --------------------------------------------------------------- write
    def _row(
        self,
        data: AcquisitionData,
        kind: str,
        group: str | None,
        local_index,
        channel,
        pattern_id=None,
        pattern_index=None,
        dmd_index=None,
    ) -> dict:
        ev = data.event
        pos = ev.position.as_dict()
        row = {
            "experiment": ev.experiment_name,
            "kind": kind,
            "t": int(ev.t_index),
            "group": group,
            "local_index": local_index,
            "channel": channel,
            "scheduled_at": _iso(ev.scheduled_time),
            "completed_at": _iso(ev.completed_time),
            "exposure_ms": float(ev.exposure_time_ms),
            "binning": int(ev.binning),
            "x": pos.get("x"),
            "y": pos.get("y"),
            "z": pos.get("z"),
            "pfs_offset": pos.get("PFSOffset", pos.get("autofocus_offset")),
            "pattern_id": None if pattern_id is None else str(pattern_id),
            "pattern_index": pattern_index,
            "dmd_index": dmd_index,
            "pixel_size_um": None
            if ev.pixel_width_um is None
            else float(ev.pixel_width_um),
        }
        # settings a pattern method changed at runtime, in force for this frame
        row.update(getattr(ev, "overrides", None) or {})
        return row

    def write_frame(self, data: AcquisitionData):
        try:
            ev = data.event
            store = self.stores.get(ev.experiment_name)
            if store is None:
                logger.warning(f"no store for experiment {ev.experiment_name}")
                return
            channel = ev.index.get("c")
            plan = self.plan

            pattern_index = None
            dmd_index = None
            pattern_id = None
            if isinstance(data, StimulationData):
                pattern_id = data.pattern_id
                pattern_index, dmd_index = self._record_pattern(
                    store, data, ev.experiment_name, ev.t_index
                )
                self.rows.append(
                    self._row(
                        data,
                        "stim_event",
                        None,
                        None,
                        channel,
                        pattern_id,
                        pattern_index,
                        dmd_index,
                    )
                )

            if ev.save_output:
                g = store.group_for(channel)
                i = None if g is None else g.local_index(ev.t_index)
                if g is None or i is None:
                    self.dropped_frames += 1
                    logger.error(
                        f"no slot for frame {ev.index} of {ev.experiment_name}; dropped "
                        f"({self.dropped_frames} so far)"
                    )
                else:
                    arr = store.images[g.name]
                    frame = np.asarray(data.data, dtype=np.uint16)
                    self._fit_group_shape(store, g, frame.shape)
                    arr[i, g.channel_index(channel)] = frame
                    store.written.add((ev.t_index, channel))
                    store.written_groups.add(g.name)
                    self.rows.append(
                        self._row(
                            data,
                            "frame",
                            g.name,
                            i,
                            channel,
                            pattern_id,
                            pattern_index,
                            dmd_index,
                        )
                    )

            # progress counts every planned frame, saved or not (a stimulation event
            # whose frame is not saved still completes its timepoint)
            self._update_progress(store, ev.experiment_name, ev.t_index)
            self._write_frames_table()
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to write frame: {e}", exc_info=True)

    def write_skipped(self, data):
        """A stimulation frame the microscope skipped: a row, and the slot counts as done."""
        try:
            ev = data.event
            store = self.stores.get(ev.experiment_name)
            if store is None:
                return
            channel = ev.index.get("c")
            self.rows.append(self._row(data, "skipped", None, None, channel))
            store.written.add((ev.t_index, channel))
            self._update_progress(store, ev.experiment_name, ev.t_index)
            self._write_frames_table()
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to record a skipped frame: {e}", exc_info=True)

    def _fit_group_shape(self, store: _ExperimentStore, g: CadenceGroup, shape):
        """
        Resize a group's arrays to the first frame's shape if the camera ROI
        reported at open time disagrees with what was delivered (the HDF5
        writer tolerated this by resizing datasets; the simulator relies on it).
        """
        arr = store.images[g.name]
        if tuple(arr.shape[2:]) == tuple(shape):
            return
        if g.name in store.written_groups:
            raise ValueError(
                f"frame shape {shape} does not match {g.name} array {arr.shape[2:]}"
            )
        logger.info(f"{g.name}: resizing arrays from {arr.shape[2:]} to {shape}")
        arr.resize((arr.shape[0], arr.shape[1], *shape))
        extras = [a for (gname, _n), a in store.labels.items() if gname == g.name]
        if g.name in store.tracks:
            extras.append(store.tracks[g.name])
        for lab in extras:
            lab.resize((lab.shape[0], lab.shape[1], *shape))

    def _record_pattern(
        self, store: _ExperimentStore, data: StimulationData, exp_name: str, t: int
    ) -> tuple[int | None, int | None]:
        """
        Store the pattern of a stimulation event according to the policy.
        Returns ``(camera index, dmd index)`` into ``patterns/camera/0`` and
        ``patterns/dmd/0``; None where nothing was stored.
        """
        if store.pattern_array is None or self.pattern_policy == "none":
            return None, None
        if self.pattern_policy == "imaging" and not any(
            ds.save and not ds.is_stim for ds in self.plan.datasets_at(exp_name, t)
        ):
            return None, None
        pid = str(data.pattern_id)

        # DMD image (one per tile for a grid experiment, contiguous in tile order)
        if isinstance(data.dmd_pattern, list | tuple):
            images = list(data.dmd_pattern)
            ids = list(data.dmd_ids or [f"{pid}:{k}" for k in range(len(images))])
            tiles = (
                [list(rc) for rc in store.geometry.tiles]
                if store.geometry is not None
                else [None] * len(images)
            )
        else:
            images, ids, tiles = [data.dmd_pattern], [pid], [None]
        if ids[0] in store.pattern_ids:
            dmd_index = store.pattern_ids.index(ids[0])
        else:
            dmd_index = None
            for image, dmd_id, tile in zip(images, ids, tiles, strict=True):
                n = self._append(store.pattern_array, image)
                dmd_index = n if dmd_index is None else dmd_index
                store.pattern_ids.append(dmd_id)
                store.dmd_camera_ids.append(pid)
                store.dmd_tiles.append(tile)
            grp = store.root["patterns/dmd"]
            meta = dict(grp.attrs.get("pyclm", {}))
            meta["pattern_ids"] = list(store.pattern_ids)
            meta["camera_ids"] = list(store.dmd_camera_ids)
            meta["tiles"] = list(store.dmd_tiles)
            grp.attrs["pyclm"] = meta

        # camera-space pattern (absent for the blank pattern before the first generate)
        camera_index = None
        if store.camera_array is not None and data.camera_pattern is not None:
            if pid in store.camera_ids:
                camera_index = store.camera_ids.index(pid)
            else:
                cam = np.asarray(data.camera_pattern, dtype=np.uint8)
                arr = store.camera_array
                if arr.shape[0] == 0 and tuple(arr.shape[1:]) != cam.shape:
                    arr.resize((0, *cam.shape))
                camera_index = self._append(arr, cam)
                store.camera_ids.append(pid)
                grp = store.root["patterns/camera"]
                meta = dict(grp.attrs.get("pyclm", {}))
                meta["pattern_ids"] = list(store.camera_ids)
                grp.attrs["pyclm"] = meta
        return camera_index, dmd_index

    @staticmethod
    def _append(arr: zarr.Array, image) -> int:
        n = arr.shape[0]
        arr.resize((n + 1, *arr.shape[1:]))
        arr[n] = np.asarray(image, dtype=np.uint8)
        return n

    def write_labels(self, data: SegmentationData):
        try:
            ev = data.event
            store = self.stores.get(ev.experiment_name)
            if store is None:
                return
            channel = ev.index.get("c")
            g = store.group_for(channel)
            if g is None:
                return
            key = (g.name, getattr(data, "name", DEFAULT_SEGMENTATION))
            if key not in store.labels:
                return
            i = g.local_index(ev.t_index)
            if i is None:
                return
            store.labels[key][i, g.channel_index(channel)] = np.asarray(
                data.data, dtype=np.uint16
            )
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to write labels: {e}", exc_info=True)

    def write_tracks(self, data):
        try:
            ev = data.event
            store = self.stores.get(ev.experiment_name)
            if store is None:
                return
            channel = ev.index.get("c")
            g = store.group_for(channel)
            if g is None or g.name not in store.tracks:
                return
            i = g.local_index(ev.t_index)
            if i is None:
                return
            labels = np.asarray(data.data, dtype=np.uint32)
            self._fit_group_shape(store, g, labels.shape)
            store.tracks[g.name][i, g.channel_index(channel)] = labels

            px = ev.pixel_width_um
            if not px:
                px = self._pixel_size_um * g.binning
            for r in data.rows:
                self.track_rows.append(
                    {
                        "experiment": ev.experiment_name,
                        "t": int(ev.t_index),
                        "channel": channel,
                        "group": g.name,
                        "local_index": int(i),
                        "track_id": int(r.track_id),
                        "label": int(r.label),
                        "y": float(r.y),
                        "x": float(r.x),
                        "y_um": float(r.y) * float(px),
                        "x_um": float(r.x) * float(px),
                        "area": int(r.area),
                        "parent": int(r.parent),
                    }
                )
            self._write_tracks_table()
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to write tracks: {e}", exc_info=True)

    def _update_progress(self, store: _ExperimentStore, exp_name: str, t: int):
        if t <= store.current_t:
            return
        for ds in self.plan.datasets_at(exp_name, t):
            if ds.save and (t, ds.channel) not in store.written:
                return
        store.current_t = t
        meta = dict(store.root.attrs["pyclm"])
        meta["current_t"] = t
        store.root.attrs["pyclm"] = meta

    def _frames_table(self) -> pa.Table:
        columns = {k: [r.get(k) for r in self.rows] for k in FRAMES_COLUMNS}
        schema = pa.schema(
            [
                ("experiment", pa.string()),
                ("kind", pa.string()),
                ("t", pa.int32()),
                ("group", pa.string()),
                ("local_index", pa.int32()),
                ("channel", pa.string()),
                ("scheduled_at", pa.string()),
                ("completed_at", pa.string()),
                ("exposure_ms", pa.float64()),
                ("binning", pa.int16()),
                ("x", pa.float64()),
                ("y", pa.float64()),
                ("z", pa.float64()),
                ("pfs_offset", pa.float64()),
                ("pattern_id", pa.string()),
                ("pattern_index", pa.int32()),
                ("dmd_index", pa.int32()),
                ("pixel_size_um", pa.float64()),
            ]
        )
        table = pa.table(columns, schema=schema)
        # one column per setting a pattern method changed during the run
        extras = sorted({k for r in self.rows for k in r if k not in FRAMES_COLUMNS})
        for key in extras:
            values = [r.get(key) for r in self.rows]
            try:
                column = pa.array(values)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                column = pa.array([None if v is None else str(v) for v in values])
            table = table.append_column(key, column)
        return table

    def _write_frames_table(self):
        if self.frames_path is None:
            return
        self._write_parquet(self._frames_table(), self.frames_path)

    def _tracks_table(self) -> pa.Table:
        columns = {k: [r.get(k) for r in self.track_rows] for k in TRACKS_COLUMNS}
        return pa.table(columns, schema=TRACKS_SCHEMA)

    def _write_tracks_table(self):
        if self.tracks_path is None:
            return
        self._write_parquet(self._tracks_table(), self.tracks_path)

    @staticmethod
    def _write_parquet(table: pa.Table, path: Path):
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        shutil.move(str(tmp), str(path))

    # --------------------------------------------------------------- close
    def close(self):
        if self.frames_path is not None and self.rows:
            try:
                self._write_frames_table()
                pacsv.write_csv(
                    self._frames_table(), self.frames_path.with_suffix(".csv")
                )
            except Exception as e:
                logger.error(f"failed to finalise frames table: {e}", exc_info=True)
        if self.tracks_path is not None and self.track_rows:
            try:
                self._write_tracks_table()
                pacsv.write_csv(
                    self._tracks_table(), self.tracks_path.with_suffix(".csv")
                )
            except Exception as e:
                logger.error(f"failed to finalise tracks table: {e}", exc_info=True)
        self.stores.clear()
        self.is_open = False
