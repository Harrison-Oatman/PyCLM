import json
import shutil
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from pyclm.convert_hdf5s import get_binning_from_metadata, make_tif
from pyclm.core.datatypes import AcquisitionData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.experiments import ExperimentSchedule, TimeCourse
from pyclm.core.manager import MicroscopeOutbox
from pyclm.core.queues import AllQueues


class MockSchedule(ExperimentSchedule):
    def __init__(self, experiment_names):
        self.experiment_names = experiment_names
        self.times = MagicMock()
        self.times.count = 3

        self.experiments = {}
        # Mock experiment config for metadata
        for name in experiment_names:
            exp = MagicMock()
            exp.experiment_name = name
            exp.t_delay = 0
            exp.t_stop = 0
            exp.as_dict.return_value = {"channels": {"638": {"binning": 2}}}
            exp.stimulation.exposure = 0

            channel = MagicMock()
            channel.every_t = 1
            channel.binning = 2
            channel.save = True
            channel.get_config_groups.return_value = []
            channel.get_device_properties.return_value = []
            exp.channels = {"638": channel}

            exp.segmentation.save = False

            self.experiments[name] = exp

    def as_dict(self):
        return {
            "experiment_names": self.experiment_names,
            "experiments": {
                name: self.experiments[name].as_dict() for name in self.experiment_names
            },
        }


class MockCore:
    def getROI(self):
        return (0, 0, 512, 512)

    def getSLMDevice(self):
        return None


def test_swmr_initialization_and_reading():
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        aq = AllQueues()
        stop_event = threading.Event()

        outbox = MicroscopeOutbox(aq, base_path=base_path, stop_event=stop_event)

        try:
            # 1. Initialize
            schedule = MockSchedule(["test_exp"])
            mock_core = MockCore()
            outbox.initialize(schedule, mock_core)

            # Verify file exists and is SWMR
            h5_path = base_path / "test_exp.hdf5"
            assert h5_path.exists()

            # 2. Write Data
            # Mock an AcquisitionEvent
            # Needs to match structure expected by convert_hdf5s: t_val points to channel group?
            # manager.py: f.create_dataset(relpath + dset_name, ...)
            # if relpath="t001/channel_638/", dset_name="data", result="t001/channel_638/data"
            # convert_hdf5s: f[t_val] -> channel_key in data
            # if t_val="t001", data=Group("t001"). "channel_638" in data -> True.

            event = MagicMock()
            event.name = "test_exp"
            event.experiment_name = "test_exp"
            event.save_output = True
            event.save_stim = False
            # Return internal_path as per new get_rel_path signature (single string)
            event.get_rel_path.return_value = "00000/channel_638/"
            event.t_index = 0

            # Write attrs mock
            def write_attrs(dset):
                dset.attrs["foo"] = "bar"

            event.write_attrs = write_attrs

            data = AcquisitionData(event, np.zeros((256, 256), dtype=np.uint16))

            # Instead of going through queue (which requires process loop), call write_data directly
            outbox.write_data(data)

            # 3. Close files before reading
            outbox.close_files()

            # Verify file structure and metadata
            with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
                assert f.swmr_mode
                assert "schedule_metadata" in f.attrs
                assert "experiment_metadata" in f.attrs

                # Verify binning read
                assert get_binning_from_metadata(f, "channel_638") == 2

            # Verify make_tif can read it
            # make_tif writes a tif file and requires valid affine transform now
            # Mock affine transform (2x3 matrix)
            mock_at = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)

            make_tif(str(h5_path), at=mock_at, chan="channel_638")

            # Output should be _patterns.tif now
            tif_path = base_path / "test_exp_channel_638_patterns.tif"
            assert tif_path.exists()

        finally:
            # 4. Clean up
            outbox.close_files()
