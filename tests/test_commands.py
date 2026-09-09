"""
Commands to a running experiment: the command files, and what the Manager
does with pause / resume, stop_run, stop_experiment, the set_* commands and
set_pattern, including the events rows they leave.
"""

import json
import threading
from time import sleep

import numpy as np
import pytest
from helpers import drain, make_experiment, make_plan, make_schedule

from pyclm.commands import Command, mark_done, pending, write_command
from pyclm.core.manager import Manager
from pyclm.core.messages import (
    AcquisitionEventMessage,
    PatternParamsResultMessage,
    UpdatePatternParamsMessage,
)
from pyclm.core.pattern_process import PatternProcess
from pyclm.core.patterns import ROI, CameraProperties, PatternMethod
from pyclm.core.queues import AllQueues
from pyclm.core.storage.events import EventLog


# ---------------------------------------------------------------- files
def test_command_files_round_trip(tmp_path):
    first = write_command(tmp_path, {"command": "pause"})
    second = write_command(
        tmp_path,
        Command(command="set_exposure", experiment="a.00", channel="545", ms=80),
    )
    assert first.parent == tmp_path / "commands"
    assert first.name < second.name
    assert json.loads(second.read_text()) == {
        "command": "set_exposure",
        "experiment": "a.00",
        "channel": "545",
        "ms": 80.0,
    }
    (tmp_path / "commands" / "zz-broken.json").write_text('{"command": "fly"}')

    found = pending(tmp_path)
    assert [c.command if c else None for _, c, _ in found] == [
        "pause",
        "set_exposure",
        None,
    ]
    assert "pause" in found[2][2]  # the allowed commands are listed

    done = mark_done(found[0][0])
    assert done.parent == tmp_path / "commands" / "done"
    assert [c.command if c else None for _, c, _ in pending(tmp_path)] == [
        "set_exposure",
        None,
    ]
    assert pending(tmp_path / "nowhere") == []


def test_command_validation_and_description():
    with pytest.raises(ValueError, match="set_exposure needs experiment, channel, ms"):
        Command(command="set_exposure")
    with pytest.raises(ValueError, match="at least one of x, y, z"):
        Command(command="set_position", experiment="a.00")
    with pytest.raises(ValueError, match="ms must be positive"):
        Command(command="set_exposure", experiment="a.00", channel="545", ms=0)
    with pytest.raises(ValueError, match="bogus"):
        Command(command="pause", bogus=1)
    cmd = Command(command="set_position", experiment="a.00", z=12.5)
    assert cmd.describe() == 'set_position {"experiment": "a.00", "z": 12.5}'
    assert [(c.kind, c.key, c.value) for c in cmd.to_changes()] == [
        ("position", "z", 12.5)
    ]
    prop = Command(
        command="set_property",
        experiment="a.00",
        channel="stimulation",
        device="Laser",
        property="Intensity",
        value=0.3,
    )
    [change] = prop.to_changes()
    assert (change.kind, change.channel, change.key) == (
        "property",
        "stimulation",
        "Laser-Intensity",
    )


# -------------------------------------------------------------- manager
def make_manager(exp, tmp_path, steps=3, interval=0.05):
    aq = AllQueues()
    manager = Manager(aq, threading.Event())
    plan = make_plan(make_schedule([exp], steps=steps, interval=interval))
    log = EventLog(tmp_path / "events.parquet")
    manager.initialize(
        plan,
        event_log=log,
        status_path=tmp_path / "status.json",
        commands_dir=tmp_path / "commands",
    )
    manager.poll_interval = 0.0
    return aq, manager, plan, log


def rows(log, kind=None):
    return [r for r in log.table().to_pylist() if kind is None or r["kind"] == kind]


def test_set_commands_go_through_apply_settings_with_source(tmp_path):
    exp = make_experiment("exp.00")
    _aq, manager, _plan, log = make_manager(exp, tmp_path)
    write_command(
        tmp_path,
        {"command": "set_exposure", "experiment": "exp.00", "channel": "545", "ms": 80},
    )
    write_command(
        tmp_path,
        {"command": "set_exposure", "experiment": "exp.00", "channel": "638", "ms": 80},
    )
    assert manager.poll_commands()

    assert exp.channels["545"].exposure == 80.0
    exposure_rows = rows(log, "exposure")
    assert [(r["status"], r["source"]) for r in exposure_rows] == [
        ("applied", "command"),
        ("refused", "command"),
    ]
    command_rows = rows(log, "command")
    assert [r["status"] for r in command_rows] == ["applied", "refused"]
    assert command_rows[0]["new"].startswith("set_exposure ")
    assert manager.commands_applied == 1
    assert not list((tmp_path / "commands").glob("*.json"))
    assert len(list((tmp_path / "commands" / "done").glob("*.json"))) == 2


def test_stop_experiment_and_pause_resume(tmp_path):
    exp = make_experiment("exp.00")
    _aq, manager, plan, log = make_manager(exp, tmp_path)
    assert plan.events_at(1)
    manager.apply_command(Command(command="stop_experiment", experiment="exp.00"))
    assert plan.events_at(1) == []
    assert plan.stopped == {"exp.00"}
    assert not manager.apply_command(
        Command(command="stop_experiment", experiment="exp.00")
    )
    assert not manager.apply_command(
        Command(command="stop_experiment", experiment="nope")
    )

    manager.start_time = 100.0
    assert manager.apply_command(Command(command="pause"))
    assert manager.paused
    sleep(0.05)
    assert manager.apply_command(Command(command="resume"))
    assert not manager.paused
    assert manager.start_time > 100.0
    assert manager.paused_s >= 0.05
    assert not manager.apply_command(Command(command="resume"))
    statuses = [r["status"] for r in rows(log, "command")]
    assert statuses == [
        "applied",
        "refused",
        "refused",
        "applied",
        "applied",
        "refused",
    ]


def test_stop_run_ends_the_loop_early(tmp_path):
    exp = make_experiment("exp.00")
    aq, manager, _plan, _log = make_manager(exp, tmp_path, steps=6, interval=0.05)
    write_command(tmp_path, {"command": "stop_run"})
    manager.process()
    dispatched = {
        m.event.t_index
        for m in drain(aq.manager_to_microscope)
        if isinstance(m, AcquisitionEventMessage)
    }
    assert dispatched == {0}  # the command is read during the wait for t = 1
    assert manager.stopping
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stopping"] is True
    assert status["commands_applied"] == 1
    # close messages still go out, so the pipeline drains normally
    assert any(m.message == "close" for m in drain(aq.manager_to_pattern))


def test_set_pattern_round_trip(tmp_path):
    class Tunable(PatternMethod):
        name = "tunable"

        def __init__(self, gain=1.0, **kwargs):
            super().__init__(**kwargs)
            self.gain = gain
            self.add_requirement("545", raw=True)

        def generate(self, context):
            return np.zeros(self.pattern_shape, np.float32)

    exp = make_experiment("exp.00", pattern_method="tunable")
    aq, manager, _plan, log = make_manager(exp, tmp_path)
    pp = PatternProcess(aq)
    pp.register_method(Tunable, "tunable")
    pp.initialize(CameraProperties(ROI(0, 0, 8, 8), 1.0))
    pp.request_method(exp)
    pp.initialize_models()

    manager.apply_command(
        Command(
            command="set_pattern",
            experiment="exp.00",
            parameters={"gain": 2.5, "bogus": 1},
        )
    )
    [msg] = [
        m
        for m in drain(aq.manager_to_pattern)
        if isinstance(m, UpdatePatternParamsMessage)
    ]
    pp.handle_message(msg)
    assert pp.models["exp.00"].gain == 2.5
    [result] = drain(aq.pattern_to_manager)
    assert isinstance(result, PatternParamsResultMessage)
    assert result.applied == {"gain": 2.5}
    assert result.refused == {"bogus": "unknown parameter"}
    manager.handle_message(result)
    pattern_rows = rows(log, "pattern")
    assert [(r["key"], r["status"], r["new"]) for r in pattern_rows] == [
        ("gain", "applied", "2.5"),
        ("bogus", "refused", None),
    ]
    assert pattern_rows[1]["detail"] == "unknown parameter"
