"""
Runtime setting changes (Stage 4): a pattern method's requests through the
context, their validation and application by the Manager at the timepoint
boundary, the events table, the override columns of the frames table, the
microscope's acknowledgements, and the status file.
"""

import json
import threading

import numpy as np
import pytest
from helpers import drain, make_experiment, make_plan, make_schedule
from test_storage import AFFINE, SLM, acquire, core_with

from pyclm.core import settings
from pyclm.core.datatypes import AcquisitionData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.experiments import MicroscopePosition
from pyclm.core.manager import Manager
from pyclm.core.messages import (
    AcquisitionEventMessage,
    EventDoneMessage,
    SettingsRequestMessage,
)
from pyclm.core.pattern_process import PatternProcess, RequestPattern
from pyclm.core.patterns import (
    ROI,
    AcquiredImageRequest,
    CameraProperties,
    DataDock,
    PatternContext,
    PatternMethod,
)
from pyclm.core.queues import AllQueues
from pyclm.core.storage import OMEZarrWriter
from pyclm.core.storage.events import EventLog


def make_manager(exp, tmp_path=None, steps=3, interval=0.02):
    aq = AllQueues()
    manager = Manager(aq, threading.Event())
    plan = make_plan(make_schedule([exp], steps=steps, interval=interval))
    log = EventLog(None if tmp_path is None else tmp_path / "events.parquet")
    manager.initialize(
        plan,
        event_log=log,
        status_path=None if tmp_path is None else tmp_path / "status.json",
        health=lambda: {"errors": {"microscope": 0}},
    )
    return aq, manager, plan, log


def four_changes():
    return [
        settings.exposure("545", 50),
        settings.config("545", "Channel", "GFP"),
        settings.device_property("stimulation", "Laser", "Intensity", 0.3),
        settings.position("z", 12.5),
    ]


# ------------------------------------------------------------------ context
def test_context_collects_requests_and_reads_back_settings():
    exp = make_experiment("exp.00")
    cid = exp.channels["545"].channel_id
    dock = DataDock(1.0, [AcquiredImageRequest(cid, True, False)])
    event = AcquisitionEvent(
        "exp.00", None, cid, index={"t": 2, "p": "exp.00", "c": "545"}
    )
    dock.add(AcquisitionData(event, np.zeros((4, 4), np.uint16)))
    position = MicroscopePosition(
        1.0, 2.0, 3.0, label="exp.00", extras={"PFSOffset": 9.0}
    )
    context = PatternContext(dock, exp, t=2, position=position)

    assert context.settings("545") == {
        "exposure_ms": 10.0,
        "binning": 1,
        "config_groups": {"Channel": "545"},
        "device_properties": {},
    }
    assert context.settings("stimulation")["config_groups"] == {"Channel": "DMD"}
    assert context.position() == {
        "label": "exp.00",
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "PFSOffset": 9.0,
    }
    assert context.requests == []

    context.set_exposure("545", 50)
    context.set_config("545", "Channel", "GFP")
    context.set_property("stimulation", "Laser", "Intensity", 0.3)
    context.set_position(z=12.5, pfs_offset=10.0)
    kinds = [(c.kind, c.channel, c.key, c.value) for c in context.requests]
    assert kinds == [
        ("exposure", "545", "exposure_ms", 50.0),
        ("config", "545", "Channel", "GFP"),
        ("property", "stimulation", "Laser-Intensity", 0.3),
        ("position", None, "z", 12.5),
        ("position", None, "pfs_offset", 10.0),
    ]
    with pytest.raises(ValueError, match="not found"):
        context.set_exposure("638", 5)


def test_pattern_process_ships_requests_to_the_manager():
    class Asks(PatternMethod):
        name = "asks"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_requirement("545", raw=True)

        def generate(self, context):
            context.set_exposure("545", 20 + context.t)
            return np.zeros(self.pattern_shape, np.float32)

    exp = make_experiment("exp.00", pattern_method="asks")
    aq = AllQueues()
    pp = PatternProcess(aq)
    pp.register_method(Asks, "asks")
    pp.initialize(CameraProperties(ROI(0, 0, 8, 8), 1.0))
    reqs = pp.request_method(exp)
    pp.initialize_models()
    pp.positions = {"exp.00": MicroscopePosition(0, 0, 0, label="exp.00")}

    pp.handle_message(RequestPattern(5, 42.0, "exp.00", reqs))
    cid = exp.channels["545"].channel_id
    event = AcquisitionEvent(
        "exp.00", None, cid, index={"t": 5, "p": "exp.00", "c": "545"}
    )
    pp.handle_data(AcquisitionData(event, np.zeros((8, 8), np.uint16)))

    [msg] = drain(aq.pattern_to_manager)
    assert isinstance(msg, SettingsRequestMessage)
    assert msg.experiment_name == "exp.00"
    assert msg.t_requested == 5
    assert [(c.kind, c.value) for c in msg.changes] == [("exposure", 25.0)]
    assert not aq.pattern_to_slm.empty()


# ------------------------------------------------------------------ manager
def test_manager_applies_requests_from_the_current_timepoint(tmp_path):
    exp = make_experiment("exp.00")
    aq, manager, plan, log = make_manager(exp, tmp_path)
    manager.current_t = 2
    manager.apply_settings(SettingsRequestMessage("exp.00", 1, four_changes()))

    assert exp.channels["545"].exposure == 50.0
    assert {g.group: g.config for g in exp.channels["545"].get_config_groups()} == {
        "Channel": "GFP"
    }
    [prop] = exp.stimulation.get_device_properties()
    assert (prop.device, prop.property, prop.value, prop.type) == (
        "Laser",
        "Intensity",
        0.3,
        "float",
    )
    assert manager.positions["exp.00"].z == 12.5
    assert manager.overrides == {
        ("exp.00", "545"): {"Channel": "GFP"},
        ("exp.00", "DMD"): {"Laser-Intensity": 0.3},
    }
    assert manager.settings_applied == 4

    rows = log.table().to_pylist()
    assert [r["status"] for r in rows] == ["applied"] * 4
    assert {(r["t_requested"], r["t_applied"]) for r in rows} == {(1, 2)}
    assert [r["old"] for r in rows] == ["10", "545", None, "0.0"]
    assert [r["channel"] for r in rows] == ["545", "545", "DMD", None]
    assert (tmp_path / "events.parquet").exists()

    # the next burst carries the new values and the override columns
    for ev in plan.events_at(2):
        manager.dispatch(ev, 0.0)
    events = [
        m.event
        for m in drain(aq.manager_to_microscope)
        if isinstance(m, AcquisitionEventMessage)
    ]
    by_channel = {e.index["c"]: e for e in events}
    assert by_channel["545"].exposure_time_ms == 50.0
    assert by_channel["545"].overrides == {"Channel": "GFP"}
    assert by_channel["545"].position.z == 12.5
    assert by_channel["DMD"].overrides == {"Laser-Intensity": 0.3}
    assert by_channel["DMD"].devices[0].value == 0.3


def test_manager_refuses_bad_requests_and_keeps_the_rest():
    exp = make_experiment("exp.00")
    _aq, manager, _plan, log = make_manager(exp)
    manager.apply_settings(
        SettingsRequestMessage(
            "exp.00",
            0,
            [
                settings.exposure("638", 5),  # unknown channel
                settings.exposure("545", -1),  # not positive
                settings.SettingChange("colour", "545", "hue", 1),  # unknown kind
                settings.position("theta", 1.0),  # unknown key
                settings.exposure("545", 7),  # fine
            ],
        )
    )
    rows = log.table().to_pylist()
    assert [r["status"] for r in rows] == ["refused"] * 4 + ["applied"]
    assert "unknown channel" in rows[0]["detail"]
    assert "positive" in rows[1]["detail"]
    assert manager.settings_refused == 4
    assert exp.channels["545"].exposure == 7.0

    manager.apply_settings(
        SettingsRequestMessage("nope", 0, [settings.exposure("545", 5)])
    )
    assert log.table().to_pylist()[-1]["detail"] == "unknown experiment 'nope'"


def test_acknowledgements_lateness_errors_and_status(tmp_path):
    exp = make_experiment("exp.00")
    _aq, manager, plan, log = make_manager(exp, tmp_path, interval=10.0)
    manager.start_time = 100.0
    cid = exp.channels["545"].channel_id

    def done(t, completed, error=None):
        event = AcquisitionEvent(
            "exp.00",
            None,
            cid,
            index={"t": t, "p": "exp.00", "c": "545"},
            scheduled_time=100.0,
        )
        event.completed_time = completed
        return EventDoneMessage(event, error=error)

    manager.handle_message(done(0, 100.5))
    manager.handle_message(done(1, 100.0 + plan.interval_s + 2.0))  # late
    manager.handle_message(
        done(1, 100.0 + plan.interval_s + 3.0)
    )  # late again, warned once
    manager.handle_message(done(2, None, error="RuntimeError('camera')"))

    assert manager.acks[0]["exp.00"]["lateness_s"] == 0.5
    assert manager.acks[2]["exp.00"]["errors"] == 1
    kinds = [r["kind"] for r in log.table().to_pylist()]
    assert kinds == ["late", "acquisition_error"]

    manager.current_t = 2
    status = manager.status()
    assert status["experiments"]["exp.00"] == {
        "last_t": 2,
        "lateness_s": 0.0,
        "errors": 1,
        "skipped": 0,
    }
    assert status["health"] == {"errors": {"microscope": 0}}
    assert status["done"] is False

    manager.finish()
    written = json.loads((tmp_path / "status.json").read_text())
    assert written["done"] is True
    assert written["t"] == 2
    assert (tmp_path / "events.csv").exists()
    assert log.closed


def test_manager_run_writes_status_and_absorbs_late_acks(tmp_path):
    exp = make_experiment("exp.00")
    aq, manager, _plan, _log = make_manager(exp, tmp_path, steps=2)
    manager.process()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["t"] == 1
    assert status["done"] is False
    # acknowledgements that arrive after the loop are absorbed by finish()
    cid = exp.channels["545"].channel_id
    event = AcquisitionEvent(
        "exp.00",
        None,
        cid,
        index={"t": 1, "p": "exp.00", "c": "545"},
        scheduled_time=1.0,
    )
    event.completed_time = 1.2
    aq.microscope_to_manager.put(EventDoneMessage(event))
    manager.finish()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["done"] is True
    assert status["experiments"]["exp.00"]["last_t"] == 1


# ------------------------------------------------------------------- frames
def test_frames_table_gains_a_column_per_changed_setting(tmp_path):
    exp = make_experiment("exp.00", every_t=1)
    exp.stimulation.save = False
    plan = make_plan(make_schedule([exp], steps=2, interval=1.0))
    writer = OMEZarrWriter()
    writer.open(plan, core_with(), tmp_path, AFFINE, SLM)
    first = acquire(plan, "exp.00", 0, "545", np.zeros((16, 16), np.uint16))
    writer.write_frame(first)
    second = acquire(plan, "exp.00", 1, "545", np.zeros((16, 16), np.uint16))
    second.event.overrides = {"LaserFarRed-Intensity": 0.4, "Channel": "GFP"}
    writer.write_frame(second)
    writer.close()

    import pyarrow.parquet as pq

    table = pq.read_table(tmp_path / "frames.parquet")
    assert {"LaserFarRed-Intensity", "Channel"} <= set(table.column_names)
    rows = {r["t"]: r for r in table.to_pylist() if r["kind"] == "frame"}
    assert rows[0]["LaserFarRed-Intensity"] is None
    assert rows[0]["Channel"] is None
    assert rows[1]["LaserFarRed-Intensity"] == 0.4
    assert rows[1]["Channel"] == "GFP"


def test_event_log_without_a_path_keeps_rows_in_memory():
    log = EventLog(None)
    log.record(
        "exposure", "exp.00", "545", "exposure_ms", 10, 20, t_requested=1, t_applied=2
    )
    assert len(log) == 1
    assert log.table().column("new").to_pylist() == ["20"]
    log.close()
    assert log.closed
