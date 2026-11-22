"""
``pyclm run --detached``: start a run in a process of its own, outside the
terminal or agent that asked for it.

A run started from a terminal belongs to that terminal's process tree. Closing
the window, ending the process tree in the Task Manager, or restarting the
agent that typed the command then stops the run partway. ``launch`` starts
``python -m pyclm run <dir> --no-check ...`` as a new process that does not
share the caller's console or process group, returns at once, and leaves a
record of what it started:

``<dir>/detached.json``
    the process id, the command, when it started and where its output goes
``<dir>/detached.log``
    the run's console output (warnings, errors, a traceback if it crashes);
    the run's own ``log.log`` is unchanged

On Windows the child gets ``DETACHED_PROCESS`` and ``CREATE_NEW_PROCESS_GROUP``,
and ``CREATE_BREAKAWAY_FROM_JOB`` where the caller's job object allows it (a
job that kills its processes when it closes would otherwise take the run with
it). Elsewhere the child starts a new session. The launching process exits
right away, so the run is not a child of anything the caller owns.

Nothing here talks to the microscope: the run in the new process does that,
exactly as ``pyclm run`` would.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RECORD_NAME = "detached.json"
LOG_NAME = "detached.log"

# Windows process creation flags (subprocess exposes them on Windows only)
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000


@dataclass
class DetachedRun:
    """What :func:`launch` started."""

    pid: int
    command: list[str]
    log: Path
    record: Path
    broke_away_from_job: bool | None  # Windows only; None elsewhere

    def text(self) -> str:
        lines = [
            f"started the run in its own process (pid {self.pid})",
            f"  console output: {self.log}",
            f"  record:         {self.record}",
        ]
        if self.broke_away_from_job is False:
            lines.append(
                "  note: this terminal's job object did not allow the run to leave it; "
                "closing the job's owner may still stop the run"
            )
        lines.append(f"follow it with: pyclm watch {self.record.parent}")
        return "\n".join(lines)


def run_command(
    directory: Path, config: str | None = None, dry: bool = False
) -> list[str]:
    """The command the detached process runs: ``pyclm run`` with the check already done."""
    cmd = [sys.executable, "-m", "pyclm", "run", str(directory), "--no-check"]
    if config:
        cmd += ["--config", str(config)]
    if dry:
        cmd.append("--dry")
    return cmd


def _popen_kwargs_windows(breakaway: bool) -> dict:
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    if breakaway:
        flags |= CREATE_BREAKAWAY_FROM_JOB
    return {"creationflags": flags, "close_fds": True}


def launch(
    directory: Path,
    config: str | None = None,
    dry: bool = False,
    *,
    popen=subprocess.Popen,
    platform: str | None = None,
) -> DetachedRun:
    """
    Start ``pyclm run`` on ``directory`` in a detached process and return at once.

    :param directory: the experiment directory
    :param config: the ``pyclm_config.toml`` to pass on, if any
    :param dry: run on the virtual microscope
    :param popen: ``subprocess.Popen`` (replaced in tests)
    :param platform: ``sys.platform`` (replaced in tests)
    """
    directory = Path(directory).resolve()
    platform = sys.platform if platform is None else platform
    cmd = run_command(directory, config, dry)
    log_path = directory / LOG_NAME
    record_path = directory / RECORD_NAME

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYCLM_DETACHED"] = "1"

    with open(log_path, "ab") as log:
        common = {
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "cwd": str(directory),
            "env": env,
        }
        broke_away: bool | None = None
        if platform == "win32":
            try:
                proc = popen(cmd, **common, **_popen_kwargs_windows(True))
                broke_away = True
            except OSError:
                # the caller's job object forbids breakaway: start inside it
                proc = popen(cmd, **common, **_popen_kwargs_windows(False))
                broke_away = False
        else:
            proc = popen(cmd, **common, start_new_session=True, close_fds=True)

    record = {
        "pid": proc.pid,
        "command": cmd,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "log": str(log_path),
        "platform": platform,
        "broke_away_from_job": broke_away,
    }
    tmp = record_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(record_path)
    return DetachedRun(proc.pid, cmd, log_path, record_path, broke_away)


def read_record(directory: Path) -> dict | None:
    """The ``detached.json`` of a directory, or None."""
    path = Path(directory) / RECORD_NAME
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def process_alive(pid: int, platform: str | None = None) -> bool | None:
    """Whether a process id is still running; None if this cannot be told."""
    platform = sys.platform if platform is None else platform
    if pid is None or pid <= 0:
        return None
    if platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True
