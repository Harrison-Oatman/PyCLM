"""
The run record (``run_record.json``) and the one-run-at-a-time microscope
lock: what a finished dry run records, how a crash is recorded, and how a
second run on the same computer is refused while a stale lock is replaced.
"""

import json
import platform
import subprocess
import sys

import pytest

from pyclm import cli
from pyclm.lock import MicroscopeLock, RunLocked, holder, lock_path
from pyclm.run_record import build_record, now


@pytest.fixture
def lock_dir(tmp_path, monkeypatch):
    d = tmp_path / "locks"
    monkeypatch.setenv("PYCLM_LOCK_DIR", str(d))
    return d


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# ------------------------------------------------------------------ lock
def test_lock_is_exclusive_and_released(lock_dir, tmp_path):
    assert lock_path().parent == lock_dir
    with MicroscopeLock(tmp_path / "run_a") as lock:
        info = json.loads(lock.path.read_text())
        assert info["directory"].endswith("run_a")
        assert holder() is not None
        with pytest.raises(RunLocked, match="run_a"):
            MicroscopeLock(tmp_path / "run_b").acquire()
    assert not lock_path().exists()
    assert holder() is None
    with MicroscopeLock(tmp_path / "run_b"):
        pass


def test_a_stale_lock_is_replaced(lock_dir, tmp_path):
    lock_dir.mkdir()
    stale = {"pid": _dead_pid(), "host": platform.node(), "directory": "old"}
    lock_path().write_text(json.dumps(stale))
    assert holder() is None
    with MicroscopeLock(tmp_path / "new") as lock:
        assert lock.replaced["directory"] == "old"
        assert json.loads(lock_path().read_text())["directory"].endswith("new")


def test_a_lock_from_another_host_is_respected(lock_dir, tmp_path):
    lock_dir.mkdir()
    other = {"pid": _dead_pid(), "host": "not-" + platform.node(), "directory": "x"}
    lock_path().write_text(json.dumps(other))
    with pytest.raises(RunLocked):
        MicroscopeLock(tmp_path / "new").acquire()


def test_detached_launch_refuses_while_locked(lock_dir, tmp_path):
    with MicroscopeLock(tmp_path / "busy"):
        code = cli.main(["run", str(tmp_path / "other"), "--detached", "--no-check"])
    assert code == 1
    assert not (tmp_path / "other" / "detached.json").exists()


# ---------------------------------------------------------------- record
@pytest.fixture(scope="module")
def recorded_demo(tmp_path_factory):
    from pyclm.demo import create_demo
    from pyclm.run_pyclm import run_pyclm

    demo = tmp_path_factory.mktemp("runs") / "demo"
    create_demo(demo)
    run_pyclm(demo, dry=True)
    return demo


def test_a_finished_run_is_recorded(recorded_demo):
    record = json.loads((recorded_demo / "run_record.json").read_text())
    assert record["outcome"] == "finished"
    assert record["error"] is None
    assert record["dry"] is True
    assert record["duration_s"] > 0
    assert record["started"][-6] in "+-"  # an offset, not a naive time
    assert record["pyclm"]["version"]
    assert len(record["config"]["pyclm_config_sha256"]) == 64
    assert record["timepoints"]["planned"] == record["timepoints"]["last_acquired"] + 1
    assert record["errors"]["total"] == 0
    exps = record["experiments"]
    assert set(exps) == {"bar.00", "cells.00"}
    assert exps["bar.00"]["toml"] == "bar.toml"
    assert exps["bar.00"]["pattern"]["method"]
    assert set(record["outputs"]) == set(exps)
    assert all(o["bytes"] > 0 for o in record["outputs"].values())


def test_a_crash_is_recorded(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"done": False}))
    record = build_record(tmp_path, started=now(), error=RuntimeError("camera timeout"))
    assert record["outcome"] == "crashed"
    assert record["error"] == "RuntimeError: camera timeout"
    (tmp_path / "status.json").write_text(json.dumps({"done": True, "stopping": True}))
    assert build_record(tmp_path, started=now())["outcome"] == "stopped"


def test_watch_flags_are_counted(tmp_path):
    (tmp_path / "watch").mkdir()
    entries = [
        {"checked": "2026-11-20T10:00:00", "flags": [{"kind": "late"}]},
        {
            "checked": "2026-11-20T10:01:00",
            "flags": [{"kind": "late"}, {"kind": "focus"}],
        },
    ]
    (tmp_path / "watch" / "flags_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries)
    )
    record = build_record(tmp_path, started=now())
    assert record["watch_flags"] == {"late": 2, "focus": 1}
