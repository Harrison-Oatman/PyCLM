"""
Skipping idle positions: when only the stimulation frame is due and the
pattern the SLM handshake delivers is blank, the microscope skips the move
and the exposure, acknowledges the acquisition as skipped, and the Manager
and the writers account for it.
"""

import json
import threading
from queue import Queue

import numpy as np
import pytest
import zarr
from helpers import FakeImageSource, make_experiment, make_plan, make_schedule
from test_dry_run import yml_experiment_dir
from test_storage import AFFINE, SLM, acquire, core_with

import pyclm.io as pio
from pyclm import run_pyclm
from pyclm.core.datatypes import EventSLMPattern, SkippedAcquisition
from pyclm.core.events import (
    AcquisitionEvent,
    UpdatePatternEvent,
    UpdateStagePositionEvent,
)
from pyclm.core.experiments import MicroscopePosition
from pyclm.core.manager import Manager
from pyclm.core.messages import EventDoneMessage
from pyclm.core.microscope import MicroscopeProcess, pattern_is_blank
from pyclm.core.patterns import AcquiredImageRequest, PatternMethod
from pyclm.core.queues import AllQueues
from pyclm.core.storage import HDF5WriterV1, OMEZarrWriter
from pyclm.core.storage.events import EventLog
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore


# ------------------------------------------------------------------ the plan
def test_plan_orders_the_handshake_first_and_flags_stimulation_only_visits():
    exp = make_experiment(
        "exp.00", every_t=2, stim_every_t=1
    )  # full_on: no requirements
    plan = make_plan(make_schedule([exp], steps=4))

    t0 = [(e.kind, e.skippable) for e in plan.events_at(0)]
    assert t0 == [
        ("request_pattern", False),
        ("update_pattern", False),
        ("position", False),
        ("acquire", False),  # stimulation
        ("acquire", False),  # imaging
    ]
    t1 = [(e.kind, e.skippable) for e in plan.events_at(1)]
    assert t1 == [
        ("request_pattern", False),
        ("update_pattern", True),
        ("position", True),
        ("acquire", True),
    ]


def test_plan_does_not_flag_a_visit_whose_stimulation_frame_the_method_needs():
    exp = make_experiment("exp.00", every_t=2, stim_every_t=1)
    needs_stim = [AcquiredImageRequest(exp.stimulation.channel_id, True, False)]
    plan = make_plan(make_schedule([exp], steps=4), {"exp.00": needs_stim})

    assert not any(e.skippable for e in plan.events_at(1))
    assert not any(e.skippable for e in plan.events_at(0))


# ------------------------------------------------------------------ the microscope
class Sink:
    def __init__(self):
        self.items = []

    def publish(self, data):
        self.items.append(data)
        return 1

    def end_stream(self, name):
        pass


def make_microscope():
    aq = AllQueues()
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))
    microscope = MicroscopeProcess(
        core, aq, stop_event=threading.Event(), settle_time_s=0.0
    )
    microscope.declare_slm()
    microscope.attach(Sink())
    return aq, microscope, core


def stim_event(exp, t, skippable):
    ev = AcquisitionEvent(
        "exp.00",
        MicroscopePosition(50.0, 0.0, 0.0, label="exp.00"),
        exp.stimulation.channel_id,
        index={"t": t, "p": "exp.00", "c": "DMD"},
        exposure_time_ms=5,
        needs_slm=True,
    )
    ev.skippable_if_blank = skippable
    return ev


def test_pattern_is_blank():
    assert pattern_is_blank(np.zeros((4, 4), np.uint8))
    assert pattern_is_blank([np.zeros((4, 4)), np.zeros((2, 2))])
    assert not pattern_is_blank(np.eye(4))
    assert not pattern_is_blank([np.zeros((4, 4)), np.eye(2)])


def test_microscope_skips_the_move_and_the_exposure_on_a_blank_pattern():
    exp = make_experiment("exp.00", every_t=2, stim_every_t=1)
    aq, microscope, core = make_microscope()
    index = {"t": 1, "p": "exp.00", "c": "DMD"}

    update = UpdatePatternEvent("exp.00", index=index, skippable_if_blank=True)
    aq.slm_to_microscope.put(
        EventSLMPattern(update.id, np.zeros((8, 8), np.uint8), "blank")
    )
    microscope.handle_update_pattern_event(update, slm_await_s=1.0)
    assert microscope.current_pattern_id == "blank"  # the blank image is still applied

    move = UpdateStagePositionEvent(
        MicroscopePosition(50.0, 0.0, 0.0),
        "exp.00",
        index=index,
        skippable_if_blank=True,
    )
    microscope.handle_update_position_event(move)
    assert core.getXYPosition() == (0.0, 0.0)  # not moved
    assert microscope.skipped_moves == 1

    microscope.handle_acquisition_event(stim_event(exp, 1, True))
    [notice] = microscope.router.items  # the writer is told on the data stream
    assert isinstance(notice, SkippedAcquisition)
    assert notice.event.t_index == 1
    microscope.router.items.clear()
    ack = aq.microscope_to_manager.get_nowait()
    assert isinstance(ack, EventDoneMessage)
    assert ack.skipped
    assert ack.t_index == 1
    assert microscope.skipped_acquisitions == 1

    # the next timepoint's pattern is lit: everything runs again
    index2 = {"t": 2, "p": "exp.00", "c": "DMD"}
    update2 = UpdatePatternEvent("exp.00", index=index2, skippable_if_blank=True)
    aq.slm_to_microscope.put(
        EventSLMPattern(update2.id, np.ones((8, 8), np.uint8), "lit")
    )
    microscope.handle_update_pattern_event(update2, slm_await_s=1.0)
    microscope.handle_update_position_event(
        UpdateStagePositionEvent(
            MicroscopePosition(50.0, 0.0, 0.0),
            "exp.00",
            index=index2,
            skippable_if_blank=True,
        )
    )
    assert core.getXYPosition() == (50.0, 0.0)
    microscope.handle_acquisition_event(stim_event(exp, 2, True))
    assert len(microscope.router.items) == 1
    assert not aq.microscope_to_manager.get_nowait().skipped


def test_microscope_never_skips_an_unflagged_event_even_when_blank():
    exp = make_experiment("exp.00", every_t=2, stim_every_t=1)
    aq, microscope, _core = make_microscope()
    index = {"t": 0, "p": "exp.00", "c": "DMD"}
    update = UpdatePatternEvent("exp.00", index=index, skippable_if_blank=False)
    aq.slm_to_microscope.put(
        EventSLMPattern(update.id, np.zeros((8, 8), np.uint8), "blank")
    )
    microscope.handle_update_pattern_event(update, slm_await_s=1.0)

    microscope.handle_acquisition_event(stim_event(exp, 0, False))
    assert len(microscope.router.items) == 1
    assert not aq.microscope_to_manager.get_nowait().skipped


# ------------------------------------------------------------------ the manager
def test_manager_records_a_skipped_acknowledgement(tmp_path):
    exp = make_experiment("exp.00", every_t=2, stim_every_t=1)
    plan = make_plan(make_schedule([exp], steps=4))
    log = EventLog(tmp_path / "events.parquet")
    manager = Manager(AllQueues(), threading.Event())
    manager.initialize(plan, event_log=log)

    manager.handle_event_done(EventDoneMessage(stim_event(exp, 1, True), skipped=True))

    assert manager.skipped == 1
    status = manager.status()
    assert status["skipped"] == 1
    assert status["experiments"]["exp.00"]["skipped"] == 1
    assert status["experiments"]["exp.00"]["last_t"] == 1
    rows = log.table().to_pylist()
    assert [(r["kind"], r["status"], r["t_applied"]) for r in rows] == [
        ("position_skipped", "skipped", 1)
    ]


# ------------------------------------------------------------------ the writers
def test_zarr_writer_counts_a_skipped_frame_as_done(tmp_path):
    exp = make_experiment("exp.00", every_t=2, stim_every_t=1)
    exp.stimulation.save = True
    plan = make_plan(make_schedule([exp], steps=4, interval=3.0))
    writer = OMEZarrWriter()
    writer.open(plan, core_with(), tmp_path, AFFINE, SLM)

    writer.write_frame(
        acquire(
            plan,
            "exp.00",
            0,
            "DMD",
            np.zeros((16, 16), np.uint16),
            np.zeros(SLM, np.uint8),
            "p0",
        )
    )
    writer.write_frame(acquire(plan, "exp.00", 0, "545", np.ones((16, 16), np.uint16)))
    skipped = acquire(plan, "exp.00", 1, "DMD", None, np.zeros(SLM, np.uint8), "p0")
    writer.write_skipped(SkippedAcquisition(skipped.event))
    writer.close()

    root = zarr.open_group(str(tmp_path / "exp.00.zarr"), mode="r")
    assert root.attrs["pyclm"]["current_t"] == 1
    data = pio.open(tmp_path / "exp.00.zarr")
    rows = data.frames.to_pylist()
    assert [(r["t"], r["kind"], r["channel"]) for r in rows if r["t"] == 1] == [
        (1, "skipped", "DMD")
    ]
    assert rows[-1]["pattern_index"] is None


def test_hdf5_writer_counts_a_skipped_frame_as_done(tmp_path):
    import h5py

    exp = make_experiment("exp.00", every_t=2, stim_every_t=1)
    exp.stimulation.save = True
    plan = make_plan(make_schedule([exp], steps=4))
    writer = HDF5WriterV1()
    writer.open(plan, core_with(), tmp_path, AFFINE, SLM)
    writer.write_frame(
        acquire(
            plan,
            "exp.00",
            0,
            "DMD",
            np.zeros((16, 16), np.uint16),
            np.zeros(SLM, np.uint8),
            "p0",
        )
    )
    writer.write_frame(acquire(plan, "exp.00", 0, "545", np.ones((16, 16), np.uint16)))
    skipped = acquire(plan, "exp.00", 1, "DMD", None, np.zeros(SLM, np.uint8), "p0")
    writer.write_skipped(SkippedAcquisition(skipped.event))
    writer.close()

    with h5py.File(tmp_path / "exp.00.hdf5", "r") as f:
        assert int(f["current_t_index"][()]) == 1
        assert bool(f["00001/stim_aq/data"].attrs["skipped"]) is True
        assert f["00001/stim_aq/data"].shape == (0, 0)


# ------------------------------------------------------------------ end to end
class Blank(PatternMethod):
    """Never lights anything."""

    name = "blank"

    def generate(self, context):
        return np.zeros(self.pattern_shape, np.float32)


def test_dry_run_skips_blank_stimulation_only_timepoints(yml_experiment_dir):
    config = yml_experiment_dir / "pyclm_config.toml"
    config.write_text(
        config.read_text().replace('format = "hdf5"', 'format = "ome-zarr"')
    )
    for name in ("bar10", "bar025"):
        toml = yml_experiment_dir / f"{name}.toml"
        text = toml.read_text()
        toml.write_text(
            text[: text.index("[pattern]")] + '[pattern]\nmethod = "blank"\n'
        )

    run_pyclm(yml_experiment_dir, dry=True, pattern_methods={"blank": Blank})

    status = json.loads((yml_experiment_dir / "status.json").read_text())
    assert status["done"] is True
    # imaging every 2 timepoints of 4: t = 1 and 3 are stimulation-only and blank
    assert status["skipped"] == 4
    for store in sorted(yml_experiment_dir.glob("*.zarr")):
        with pio.open(store) as exp:
            assert exp.current_t == 3
            kinds = {
                (r["t"], r["kind"])
                for r in exp.frames.to_pylist()
                if r["channel"] == "DMD"
            }
            assert (1, "skipped") in kinds
            assert (3, "skipped") in kinds
            assert (0, "frame") in kinds
            assert (2, "frame") in kinds
            events = [
                e for e in exp.events.to_pylist() if e["kind"] == "position_skipped"
            ]
            assert sorted(e["t_applied"] for e in events) == [1, 3]
