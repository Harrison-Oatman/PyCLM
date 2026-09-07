"""
Outbox HDF5 output: SWMR initialisation from a plan, one frame written through
the same path the microscope uses, and readability by the converter.
"""

import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from helpers import make_experiment, make_plan, make_schedule

from pyclm.convert_hdf5s import get_binning_from_metadata, make_tif
from pyclm.core.datatypes import AcquisitionData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.manager import MicroscopeOutbox
from pyclm.core.queues import AllQueues


class MockCore:
    def getROI(self):
        return (0, 0, 512, 512)

    def getSLMDevice(self):
        return None


def test_swmr_initialization_and_reading():
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        aq = AllQueues()
        outbox = MicroscopeOutbox(aq, base_path=base_path, stop_event=threading.Event())

        exp = make_experiment("test_exp", channel="638", binning=2, stim_exposure=0)
        schedule = make_schedule([exp], steps=3)
        plan = make_plan(schedule)

        try:
            outbox.initialize(plan, MockCore())

            h5_path = base_path / "test_exp.hdf5"
            assert h5_path.exists()

            # write one frame the way the microscope would deliver it
            [acquire] = [e for e in plan.events_at(0) if e.kind == "acquire"]
            cfg = plan.imaging_config("test_exp", "638")
            event = AcquisitionEvent(
                "test_exp",
                schedule.positions["test_exp"],
                cfg.channel_id,
                index=acquire.index,
                binning=2,
            )
            outbox.write_data(
                AcquisitionData(event, np.zeros((256, 256), dtype=np.uint16))
            )
            outbox.close_files()

            with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
                assert f.swmr_mode
                assert "schedule_metadata" in f.attrs
                assert "experiment_metadata" in f.attrs
                assert "pyclm" in f.attrs["plan"]
                assert f.attrs["plan_format"] == 1

                dset = f["00000/channel_638/data"]
                assert dset.shape == (256, 256)
                assert json.loads(dset.attrs["index"]) == {
                    "t": 0,
                    "p": "test_exp",
                    "c": "638",
                }
                # the only saved frame at t=0 is written, so t=0 is complete
                assert int(f["current_t_index"][()]) == 0
                # unwritten timepoints stay pre-allocated and empty
                assert f["00001/channel_638/data"].shape == (0, 0)

                assert get_binning_from_metadata(f, "channel_638") == 2

            mock_at = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
            make_tif(str(h5_path), at=mock_at, chan="channel_638")
            assert (base_path / "test_exp_channel_638_patterns.tif").exists()

        finally:
            outbox.close_files()
