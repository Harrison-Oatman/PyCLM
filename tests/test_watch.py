"""
`pyclm run --detached` and `pyclm watch`: the detached process outlives its
launcher and records itself; the watcher flags stale, late, failing and
stopped runs from the run's own files, notifies once per quiet period, and
runs --then only after the run is done and settled.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

from pyclm import cli, detach
from pyclm.watch import Flag, Watcher, WatchSettings, sharpness, watch


def _status(path, **kw):
    st = {
        "t": 3,
        "acquired_t": 3,
        "timepoints": 10,
        "done": False,
        "experiments": {"bar": {"last_t": 3, "lateness_s": 1.0, "errors": 0}},
        "health": {"errors": {}, "undeliverable": 0, "dropped_frames": 0},
    }
    st.update(kw)
    (path / "status.json").write_text(json.dumps(st))
    return st


def _kinds(flags):
    return sorted(f.kind for f in flags)


def test_no_status_then_ok(tmp_path):
    w = Watcher(tmp_path, WatchSettings(late_s=30))
    assert _kinds(w.check()) == ["no_status"]
    _status(tmp_path)
    assert w.check() == []


def test_stale_late_errors_and_log(tmp_path):
    _status(tmp_path)
    w = Watcher(tmp_path, WatchSettings(late_s=30, stale_min=10))
    assert w.check() == []
    _status(
        tmp_path,
        experiments={"bar": {"last_t": 4, "lateness_s": 95.0, "errors": 2}},
        health={"errors": {"segmentation": 1}, "undeliverable": 0, "dropped_frames": 3},
    )
    (tmp_path / "log.log").write_text(
        "10:00:00 - microscope.py - INFO     - image acquired\n"
        "10:00:01 - microscope.py - ERROR    - ZDrive: error (-9)\n"
        "10:00:02 - microscope.py - ERROR    - ZDr"  # half-written line waits
    )
    flags = w.check()
    assert _kinds(flags) == [
        "acquisition_error",
        "dropped_frames",
        "late",
        "log_error",
        "process_error",
    ]
    log_flag = next(f for f in flags if f.kind == "log_error")
    assert "1 new error line" in log_flag.message
    # nothing new: counts that did not rise are not flagged again; lateness still is
    assert _kinds(w.check()) == ["late"]
    # the completed half line is read once
    with open(tmp_path / "log.log", "a") as f:
        f.write("ive: error (-9)\n")
    assert "log_error" in _kinds(w.check())
    # stale: status.json not rewritten for 11 minutes
    assert "stale" in _kinds(w.check(now=time.time() + 11 * 60))


def test_late_defaults_to_one_interval(tmp_path):
    (tmp_path / "schedule.toml").write_text(
        "[timing]\nsteps = 10\ninterval_seconds = 60.0\nsetup_time_seconds = 0.0\n"
        "time_between_positions = 0.1\n"
    )
    _status(
        tmp_path, experiments={"bar": {"last_t": 4, "lateness_s": 61.0, "errors": 0}}
    )
    assert _kinds(Watcher(tmp_path).check()) == ["late"]


def test_stopped_detached_process(tmp_path):
    _status(tmp_path)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    (tmp_path / "detached.json").write_text(json.dumps({"pid": proc.pid}))
    assert _kinds(Watcher(tmp_path).check()) == ["stopped"]
    (tmp_path / "detached.json").write_text(json.dumps({"pid": os.getpid()}))
    assert Watcher(tmp_path).check() == []


def test_record_and_quiet_period(tmp_path):
    w = Watcher(tmp_path, WatchSettings(quiet_min=30))
    flags = [Flag("late", "bar", "x")]
    path = w.record(flags)
    assert json.loads(path.read_text())["flags"][0]["kind"] == "late"
    w.record([])
    assert len((tmp_path / "watch" / "flags_log.jsonl").read_text().splitlines()) == 2
    now = 1000.0
    assert w.to_notify(flags, now) == flags
    assert w.to_notify(flags, now + 60) == []
    assert w.to_notify(flags, now + 31 * 60) == flags


def test_then_waits_for_settled_files(tmp_path):
    _status(tmp_path, done=True)
    w = Watcher(tmp_path, WatchSettings(settle_s=30))
    w.check()
    now = time.time()
    assert not w.settled(now)
    assert not w.settled(now + 10)
    assert w.settled(now + 31)


def test_watch_once_and_then(tmp_path, capsys):
    _status(tmp_path, done=True)
    marker = tmp_path / "then_ran.txt"
    code = watch(tmp_path, WatchSettings(late_s=30), once=True)
    assert code == 0
    then = f"\"{sys.executable}\" -c \"open(r'{marker}', 'w').write('x')\""
    code = watch(tmp_path, WatchSettings(then=then, settle_s=0.0, every_s=0.01))
    assert code == 0
    assert marker.exists()


def test_image_flags(tmp_path, monkeypatch):
    from pyclm import io as pyclm_io

    rng = np.random.default_rng(0)
    sharp = rng.integers(1000, 3000, (64, 64)).astype(np.uint16)
    frames = {0: sharp, 1: sharp}

    class Group:
        channels = ("545",)

        def acquired(self):
            return sorted(frames)

        def frame(self, i, channel):
            return frames[i]

    class Exp:
        name = "bar.00"

        def __init__(self):
            self.groups = {"545": Group()}

        def close(self):
            pass

    (tmp_path / "bar.00.zarr").mkdir()
    monkeypatch.setattr(
        pyclm_io, "find_experiments", lambda d: [tmp_path / "bar.00.zarr"]
    )
    monkeypatch.setattr(pyclm_io, "open", lambda p: Exp())
    _status(tmp_path)
    w = Watcher(tmp_path, WatchSettings(images=True, late_s=30))
    assert w.check() == []
    blurred = np.full((64, 64), 400, np.uint16)  # darker and featureless
    frames[2] = blurred
    assert _kinds(w.check()) == ["brightness", "focus"]
    frames[3] = np.full((64, 64), 65535, np.uint16)
    assert "saturation" in _kinds(w.check())
    assert sharpness(sharp) > 0


# ------------------------------------------------------------- detached runs
def test_launch_records_and_windows_flags(tmp_path):
    seen = []

    class FakeProc:
        pid = 4242

    def popen(cmd, **kw):
        seen.append(kw)
        if kw["creationflags"] & detach.CREATE_BREAKAWAY_FROM_JOB:
            raise PermissionError("job does not allow breakaway")
        return FakeProc()

    run = detach.launch(tmp_path, "cfg.toml", dry=True, popen=popen, platform="win32")
    assert run.pid == 4242
    assert run.broke_away_from_job is False
    assert len(seen) == 2
    assert seen[1]["creationflags"] & detach.DETACHED_PROCESS
    assert seen[1]["creationflags"] & detach.CREATE_NEW_PROCESS_GROUP
    record = detach.read_record(tmp_path)
    assert record["pid"] == 4242
    assert record["command"][-4:] == ["--no-check", "--config", "cfg.toml", "--dry"]
    assert "job object" in run.text()


def test_detached_refuses_gui_and_check_errors(tmp_path, capsys):
    assert cli.main(["run", str(tmp_path), "--detached", "--gui"]) == 1
    # an empty directory fails the check, so nothing starts
    assert cli.main(["run", str(tmp_path), "--detached"]) == 1
    assert "not started" in capsys.readouterr().out
    assert not (tmp_path / "detached.json").exists()


def test_detached_dry_run_end_to_end(tmp_path):
    """The demo runs on the virtual microscope in its own process; the watcher sees it finish."""
    from pyclm.demo import create_demo

    demo = tmp_path / "demo"
    create_demo(demo)
    started = subprocess.run(
        [sys.executable, "-m", "pyclm", "run", str(demo), "--detached", "--dry"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert started.returncode == 0, started.stdout + started.stderr
    record = detach.read_record(demo)
    assert record
    assert record["pid"] > 0
    marker = tmp_path / "done.txt"
    then = f"\"{sys.executable}\" -c \"open(r'{marker}', 'w').write('x')\""
    code = watch(demo, WatchSettings(every_s=1.0, then=then, settle_s=2.0, late_s=1e9))
    assert code == 0
    assert marker.exists()
    status = json.loads((demo / "status.json").read_text())
    assert status["done"] is True
    assert (demo / "watch" / "flags.json").exists()
