"""
OME-Zarr (NGFF 0.4, zarr v2) writer: PyCLM storage format 2.

Layout per experiment (``<experiment>.zarr/``):

    .zattrs                     pyclm: {format: 2, plan, experiment, schedule, affine_transform,
                                slm_shape, groups: {...}, current_t, useq_version}
    <group>/                    one NGFF image per cadence group ("imaging", "stim", ...)
        .zattrs                 multiscales (axes t,c,y,x; time scale = every_t * interval), omero
        0                       (T, C, Y, X) uint16, chunks (1, 1, Y, X), zstd; compact T
        labels/segmentation/0   (T, C, Y, X) uint16 label images (only if segmentation.save)
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
from ..plan import AcquisitionPlan
from .base import (
    PATTERN_POLICIES,
    CadenceGroup,
    FrameWriter,
    cadence_groups,
    image_shape,
)

logger = logging.getLogger(__name__)

STORAGE_FORMAT_VERSION = 2
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
    "pixel_size_um",
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
        self.labels: dict[str, zarr.Array] = {}
        self.pattern_array: zarr.Array | None = None
        self.pattern_ids: list[str] = []
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
        self.dropped_frames = 0
        self.error_count = 0
        self.frames_path: Path | None = None
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
    def open(self, plan, core, base_path, affine_transform=None, slm_shape=None):
        self.plan = plan
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.base_path / "frames.parquet"
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
        root = zarr.open_group(str(path), mode="w", zarr_format=2)
        store = _ExperimentStore(root, groups)
        layers = []

        for g in groups:
            h, w = image_shape(core, g.binning)
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
            if experiment.segmentation.save:
                labels = img_group.create_group("labels")
                labels.attrs["labels"] = ["segmentation"]
                seg = labels.create_group("segmentation")
                seg.attrs.update(
                    ngff_image_attrs(
                        f"{exp_name}/{g.name}/segmentation",
                        [g.every_t * plan.interval_s, 1.0, px, px],
                        g.channels,
                    )
                )
                seg.attrs["image-label"] = {
                    "version": NGFF_VERSION,
                    "source": {"image": "../../"},
                }
                store.labels[g.name] = seg.create_array(
                    "0",
                    shape=(g.timepoints, len(g.channels), h, w),
                    chunks=(1, 1, h, w),
                    dtype="uint16",
                    compressors=_compressor(),
                    fill_value=0,
                )
            for c in g.channels:
                layers.append((str(path.resolve()), f"{g.name}/{c}"))

        if slm_shape is not None and self.pattern_policy != "none":
            patterns = root.create_group("patterns/dmd")
            patterns.attrs["pyclm"] = {"policy": self.pattern_policy, "pattern_ids": []}
            store.pattern_array = patterns.create_array(
                "0",
                shape=(0, int(slm_shape[0]), int(slm_shape[1])),
                chunks=(1, int(slm_shape[0]), int(slm_shape[1])),
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
            "pattern_policy": self.pattern_policy,
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
    ) -> dict:
        ev = data.event
        pos = ev.position.as_dict()
        return {
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
            "pixel_size_um": None
            if ev.pixel_width_um is None
            else float(ev.pixel_width_um),
        }

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
            pattern_id = None
            if isinstance(data, StimulationData):
                pattern_id = data.pattern_id
                pattern_index = self._record_pattern(store, data)
                self.rows.append(
                    self._row(
                        data,
                        "stim_event",
                        None,
                        None,
                        channel,
                        pattern_id,
                        pattern_index,
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
                            data, "frame", g.name, i, channel, pattern_id, pattern_index
                        )
                    )

            # progress counts every planned frame, saved or not (a stimulation event
            # whose frame is not saved still completes its timepoint)
            self._update_progress(store, ev.experiment_name, ev.t_index)
            self._write_frames_table()
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to write frame: {e}", exc_info=True)

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
        if g.name in store.labels:
            lab = store.labels[g.name]
            lab.resize((lab.shape[0], lab.shape[1], *shape))

    def _record_pattern(
        self, store: _ExperimentStore, data: StimulationData
    ) -> int | None:
        if store.pattern_array is None or self.pattern_policy == "none":
            return None
        pid = str(data.pattern_id)
        if self.pattern_policy == "on_change" and pid in store.pattern_ids:
            return store.pattern_ids.index(pid)
        pattern = np.asarray(data.dmd_pattern, dtype=np.uint8)
        arr = store.pattern_array
        n = arr.shape[0]
        arr.resize((n + 1, *arr.shape[1:]))
        arr[n] = pattern
        store.pattern_ids.append(pid)
        patterns_group = store.root["patterns/dmd"]
        meta = dict(patterns_group.attrs.get("pyclm", {}))
        meta["pattern_ids"] = list(store.pattern_ids)
        patterns_group.attrs["pyclm"] = meta
        return n

    def write_labels(self, data: SegmentationData):
        try:
            ev = data.event
            store = self.stores.get(ev.experiment_name)
            if store is None:
                return
            channel = ev.index.get("c")
            g = store.group_for(channel)
            if g is None or g.name not in store.labels:
                return
            i = g.local_index(ev.t_index)
            if i is None:
                return
            store.labels[g.name][i, g.channel_index(channel)] = np.asarray(
                data.data, dtype=np.uint16
            )
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to write labels: {e}", exc_info=True)

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
                ("pixel_size_um", pa.float64()),
            ]
        )
        return pa.table(columns, schema=schema)

    def _write_frames_table(self):
        if self.frames_path is None:
            return
        table = self._frames_table()
        tmp = self.frames_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        shutil.move(str(tmp), str(self.frames_path))

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
        self.stores.clear()
        self.is_open = False
