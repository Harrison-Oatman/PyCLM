"""
The original PyCLM HDF5 layout (format 1), kept for compatibility.

One SWMR file per experiment, one pre-allocated 2-D dataset per
``{t:05d}/<group>/{data,seg,dmd}``, per-dataset attributes. This is exactly
what ``MicroscopeOutbox`` wrote before Stage 2; only the home moved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import sleep

import numpy as np
from h5py import File

from ..core_interface import MicroscopeCoreInterface
from ..datatypes import AcquisitionData, SegmentationData, StimulationData
from ..events import storage_group
from ..experiments import ImagingConfig
from ..kinds import DEFAULT_SEGMENTATION
from ..plan import PLAN_FORMAT, AcquisitionPlan
from .base import FrameWriter, image_shape

logger = logging.getLogger(__name__)


class HDF5WriterV1(FrameWriter):
    format = "hdf5"

    def __init__(self):
        super().__init__()
        self.open_files: dict[str, File] = {}
        self.experiments = {}
        self.dropped_frames = 0
        self.error_count = 0

    def planned_paths(self, names, base_path):
        return {name: Path(base_path) / f"{name}.hdf5" for name in names}

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
        self.base_path = Path(base_path)
        schedule = plan.schedule
        metadata = schedule.as_dict()
        plan_yaml = plan.yaml_str()

        try:
            for exp_name in plan.experiments:
                filepath = self.base_path / f"{exp_name}.hdf5"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if filepath.exists():
                    raise FileExistsError(
                        f"HDF5 file with this name already exists: {filepath}"
                    )

                f = File(filepath, "w", libver="latest")

                experiment = schedule.experiments[exp_name]
                self.experiments[exp_name] = experiment

                f.attrs["schedule_metadata"] = json.dumps(metadata, default=str)
                f.attrs["experiment_metadata"] = json.dumps(
                    experiment.as_dict(), default=str
                )
                f.attrs["plan"] = plan_yaml
                f.attrs["plan_format"] = PLAN_FORMAT
                if routing is not None:
                    f.attrs["routing"] = json.dumps(routing, default=str)
                if affine_transform is not None:
                    f.attrs["affine_transform"] = np.asarray(
                        affine_transform, dtype=float
                    )

                f.create_dataset("current_t_index", data=np.int32(-1))

                stim_name = plan.stim_channel(exp_name)
                every_t_map = {
                    storage_group(c, c == stim_name): plan.every_t(exp_name, c)
                    for c in plan.channels(exp_name)
                }
                f.attrs["every_t"] = json.dumps(every_t_map)
                f.attrs["t_delay"] = experiment.t_delay
                f.attrs["t_stop"] = experiment.t_stop
                f.attrs["t_count"] = plan.timepoints

                self.open_files[exp_name] = f

            slm_device = core.getSLMDevice()
            dmd_shape = None
            if slm_device:
                dmd_shape = (
                    core.getSLMHeight(slm_device),
                    core.getSLMWidth(slm_device),
                )

            for ds in plan.expected_datasets():
                f = self.open_files[ds.experiment]
                experiment = self.experiments[ds.experiment]
                shape = image_shape(core, ds.config.binning)
                prefix = f"{ds.t:05d}/{ds.group}"

                self._create_frame_dataset(
                    f, f"{prefix}/data", shape, np.uint16, ds.config
                )
                if experiment.segmentation.save:
                    self._create_frame_dataset(
                        f, f"{prefix}/seg", shape, np.uint16, ds.config
                    )
                if ds.is_stim and dmd_shape is not None:
                    self._create_frame_dataset(
                        f, f"{prefix}/dmd", dmd_shape, np.uint8, ds.config
                    )

            # Enable SWMR only after all datasets exist
            for exp_name, f in self.open_files.items():
                f.swmr_mode = True
                logger.info(f"Initialized HDF5 file for {exp_name} in SWMR mode.")

        except Exception as e:
            logger.error(f"Failed to initialize outbox files: {e}", exc_info=True)
            self.close()
            raise e

        self.is_open = True

        all_layers = []
        for exp_name, experiment in schedule.experiments.items():
            filepath = str((self.base_path / f"{exp_name}.hdf5").resolve())
            for channel_name in experiment.channels:
                all_layers.append((filepath, f"channel_{channel_name}"))
            if experiment.stimulation.save:
                all_layers.append((filepath, "stim_aq"))
        return all_layers

    def _create_frame_dataset(self, f, path, maxshape, dtype, config: ImagingConfig):
        dset = f.create_dataset(
            path, shape=(0, 0), maxshape=maxshape, dtype=dtype, chunks=True
        )
        self._preallocate_attrs(dset, config)

    def _preallocate_attrs(self, dset, channel: ImagingConfig):
        dset.attrs["id"] = ""
        dset.attrs["position"] = [("", "")]
        dset.attrs["experiment_name"] = ""
        dset.attrs["time_scheduled"] = ""
        dset.attrs["time_since_start"] = ""
        dset.attrs["time_completed"] = ""
        dset.attrs["complete"] = False
        dset.attrs["exposure_time_ms"] = 0.0
        dset.attrs["needs_slm"] = False
        dset.attrs["binning"] = 1
        dset.attrs["index"] = ""
        dset.attrs["save_output"] = False
        dset.attrs["channel_id"] = ""
        dset.attrs["pixel_width_um"] = ""
        dset.attrs["pattern_id"] = ""

        for cg in channel.get_config_groups():
            dset.attrs[f"config_groups: {cg.group}"] = ""
        for dp in channel.get_device_properties():
            dset.attrs[f"devices: {dp.device}-{dp.property}"] = ""

    # --------------------------------------------------------------- write
    def write_frame(self, data: AcquisitionData):
        self._write(data, "data")

    def write_labels(self, data: SegmentationData):
        name = getattr(data, "name", DEFAULT_SEGMENTATION)
        if name != DEFAULT_SEGMENTATION:
            # format 1 has one seg dataset per frame; named segmentations need format 2
            if not getattr(self, "_warned_named_seg", False):
                self._warned_named_seg = True
                logger.warning(
                    f"HDF5 format 1 stores only the default segmentation; dropping "
                    f'the {name!r} labels (use [output] format = "ome-zarr")'
                )
            return
        self._write(data, "seg")

    def _timepoint_complete(self, f, t_index: int, exp_name: str) -> bool:
        for ds in self.plan.datasets_at(exp_name, t_index):
            if not ds.save:
                continue
            path = f"{ds.t:05d}/{ds.group}/data"
            try:
                if f[path].shape == (0, 0):
                    return False
            except KeyError:
                return False
        return True

    @staticmethod
    def _retry(fn):
        for _attempt in range(3):
            try:
                fn()
                return
            except PermissionError:
                sleep(0.05)

    def _write(self, data: AcquisitionData, dset_name: str):
        aq_event = data.event
        relpath = aq_event.get_rel_path()
        exp_name = aq_event.experiment_name

        try:
            f = self.open_files.get(exp_name)
            if f is None:
                logger.warning(f"No open file found for experiment: {exp_name}")
                return

            if aq_event.save_output:
                if (relpath + dset_name) not in f:
                    self.dropped_frames += 1
                    logger.error(
                        f"no pre-allocated dataset {relpath + dset_name} in "
                        f"{exp_name}.hdf5; frame dropped "
                        f"({self.dropped_frames} dropped so far)"
                    )
                    return

                dset = f[relpath + dset_name]
                if dset.shape != data.data.shape:
                    dset.resize(data.data.shape)

                def put():
                    dset[...] = data.data
                    aq_event.write_attrs(dset)
                    f.flush()

                self._retry(put)

            if isinstance(data, StimulationData) and dset_name == "data":
                if (relpath + "dmd") in f:
                    dmd = f[relpath + "dmd"]
                    if dmd.shape != data.dmd_pattern.shape:
                        dmd.resize(data.dmd_pattern.shape)

                    def put_dmd():
                        dmd[...] = data.dmd_pattern
                        dmd.attrs["pattern_id"] = str(data.pattern_id)
                        aq_event.write_attrs(dmd)
                        f.flush()

                    self._retry(put_dmd)

            t_index = aq_event.t_index
            if f["current_t_index"][()] < t_index and self._timepoint_complete(
                f, t_index, exp_name
            ):

                def bump():
                    f["current_t_index"][...] = np.int32(t_index)
                    f.flush()

                self._retry(bump)

        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to write data: {e}", exc_info=True)

    # --------------------------------------------------------------- close
    def close(self):
        for name, f in self.open_files.items():
            try:
                f.close()
                logger.info(f"Closed HDF5 file for {name}")
            except Exception as e:
                logger.error(f"Error closing file for {name}: {e}")
        self.open_files.clear()
        self.is_open = False
