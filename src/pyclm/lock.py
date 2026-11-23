"""
One run at a time on a microscope.

``pyclm run`` on hardware takes a lock file before it touches the
microscope and removes it when the run ends, so that a second ``pyclm run``
on the same computer (another terminal, another person's agent, a second
``--detached`` launch) refuses to start instead of fighting over the stage,
camera and DMD. Dry runs take no lock.

The lock is one file per computer, ``microscope.lock``, in a folder every
user of the computer can write: ``%PROGRAMDATA%\\pyclm`` on Windows, the
system temporary folder elsewhere, or ``PYCLM_LOCK_DIR`` when it is set. It
records who holds it (process id, host, user, directory, start time). A lock
whose process has ended (a crash, a killed terminal) is stale: the next run
replaces it and says so.
"""

from __future__ import annotations

import datetime
import getpass
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

LOCK_NAME = "microscope.lock"


class RunLocked(RuntimeError):
    """Another run holds the microscope."""

    def __init__(self, holder: dict, path: Path):
        self.holder = holder
        self.path = path
        who = holder.get("user", "?")
        since = holder.get("started", "?")
        where = holder.get("directory", "?")
        super().__init__(
            f"the microscope is in use: a run of {where} started {since} by {who} "
            f"(process {holder.get('pid', '?')}). Wait for it to finish, or stop it. "
            f"If no run is going, delete {path}."
        )


def lock_dir() -> Path:
    """The folder holding the lock file."""
    env = os.environ.get("PYCLM_LOCK_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("PROGRAMDATA") or tempfile.gettempdir()
        return Path(base) / "pyclm"
    return Path(tempfile.gettempdir()) / "pyclm-locks"


def lock_path() -> Path:
    return lock_dir() / LOCK_NAME


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "?"


def holder(path: Path | None = None) -> dict | None:
    """Who holds the lock, or None when it is free or stale."""
    from .detach import process_alive

    path = lock_path() if path is None else Path(path)
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"pid": None, "note": "unreadable lock file"}
    same_host = info.get("host") == platform.node()
    if same_host and process_alive(info.get("pid")) is False:
        return None
    return info


class MicroscopeLock:
    """
    ``with MicroscopeLock(directory): ...`` holds the microscope for the
    block; raises :class:`RunLocked` when another live run holds it.
    """

    def __init__(self, directory, path: Path | None = None):
        self.directory = Path(directory)
        self.path = lock_path() if path is None else Path(path)
        self.replaced: dict | None = None  # a stale lock that was taken over
        self._held = False

    def acquire(self) -> MicroscopeLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "pid": os.getpid(),
            "host": platform.node(),
            "user": _user(),
            "directory": str(self.directory.resolve()),
            "started": datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
        }
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                current = holder(self.path)
                if current is not None:
                    raise RunLocked(current, self.path) from None
                # stale: its process has ended
                try:
                    self.replaced = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self.replaced = {}
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=1)
            self._held = True
            return self
        raise RunLocked(holder(self.path) or {}, self.path)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if info.get("pid") == os.getpid() and info.get("host") == platform.node():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> MicroscopeLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
