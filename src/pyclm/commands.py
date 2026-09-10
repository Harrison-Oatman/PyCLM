"""
Commands to a running experiment.

A command is one small JSON file in ``<experiment directory>/commands/``,
written atomically (a temporary name, then a rename). The Manager lists the
directory at the timepoint boundary, applies each command in name order,
records it in ``events.parquet`` and moves the file to ``commands/done/``.
Anything can write one: a script, a person with an
editor::

    {"command": "set_exposure", "experiment": "bar10.00", "channel": "545", "ms": 80}

None of the commands changes the structure of the plan; ``stop_experiment``
ends one experiment early, which the storage handles as skipped frames.
See docs/stage5b-interactivity-design.md §6.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .core import settings as runtime_settings
from .core.settings import STIMULATION, SettingChange

COMMANDS_DIR = "commands"
DONE_DIR = "done"
COMMAND_NAMES = (
    "pause",
    "resume",
    "stop_run",
    "stop_experiment",
    "set_exposure",
    "set_config",
    "set_property",
    "set_position",
    "set_pattern",
)
# which fields each command needs
REQUIRED = {
    "pause": (),
    "resume": (),
    "stop_run": (),
    "stop_experiment": ("experiment",),
    "set_exposure": ("experiment", "channel", "ms"),
    "set_config": ("experiment", "channel", "group", "preset"),
    "set_property": ("experiment", "channel", "device", "property", "value"),
    "set_position": ("experiment",),
    "set_pattern": ("experiment", "parameters"),
}

_counter = [0]


class Command(BaseModel):
    """One command; ``describe()`` is what the events table records."""

    model_config = ConfigDict(extra="forbid")

    command: Literal[COMMAND_NAMES]
    experiment: str | None = None
    channel: str | None = None
    ms: float | None = None
    group: str | None = None
    preset: str | None = None
    device: str | None = None
    property: str | None = None
    value: bool | int | float | str | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    pfs_offset: float | None = None
    parameters: dict[str, Any] | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _required(self):
        missing = [f for f in REQUIRED[self.command] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.command} needs {', '.join(missing)}")
        if self.command == "set_position" and all(
            getattr(self, k) is None for k in ("x", "y", "z", "pfs_offset")
        ):
            raise ValueError("set_position needs at least one of x, y, z, pfs_offset")
        if self.command == "set_exposure" and not (self.ms and self.ms > 0):
            raise ValueError("ms must be positive")
        return self

    def describe(self) -> str:
        fields = {
            k: v
            for k, v in self.model_dump().items()
            if v is not None and k not in ("command", "note")
        }
        text = self.command
        if fields:
            text += " " + json.dumps(fields, default=str)
        return text

    def to_changes(self) -> list[SettingChange]:
        """The setting changes of a ``set_*`` command (``set_pattern`` has none)."""
        ch = self.channel if self.channel != STIMULATION else STIMULATION
        if self.command == "set_exposure":
            return [runtime_settings.exposure(ch, self.ms)]
        if self.command == "set_config":
            return [runtime_settings.config(ch, self.group, self.preset)]
        if self.command == "set_property":
            return [
                runtime_settings.device_property(
                    ch, self.device, self.property, self.value
                )
            ]
        if self.command == "set_position":
            return [
                runtime_settings.position(key, getattr(self, key))
                for key in ("x", "y", "z", "pfs_offset")
                if getattr(self, key) is not None
            ]
        return []


def commands_dir(directory) -> Path:
    return Path(directory) / COMMANDS_DIR


def write_command(directory, command: Command | dict) -> Path:
    """Write one command file atomically; returns its path."""
    if not isinstance(command, Command):
        command = Command.model_validate(command)
    folder = commands_dir(directory)
    folder.mkdir(parents=True, exist_ok=True)
    _counter[0] += 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # sorts by wall-clock time across writers, then by order within this process
    name = f"{stamp}-{time.time_ns():020d}-{_counter[0]:04d}.json"
    tmp = folder / f".{name}.tmp"
    tmp.write_text(
        command.model_dump_json(exclude_none=True, indent=1), encoding="utf-8"
    )
    final = folder / name
    os.replace(tmp, final)
    return final


def pending(directory) -> list[tuple[Path, Command | None, str | None]]:
    """Command files not yet processed in ``<directory>/commands``, in name order."""
    return pending_in(commands_dir(directory))


def pending_in(folder: Path) -> list[tuple[Path, Command | None, str | None]]:
    """Command files in ``folder``: (path, command or None, problem or None), in name order."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(p for p in folder.glob("*.json") if not p.name.startswith(".")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append((path, Command.model_validate(data), None))
        except (OSError, ValueError, ValidationError) as e:
            out.append((path, None, _problem(e)))
    return out


def _problem(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(err.get("msg", "") for err in e.errors())
    return f"{type(e).__name__}: {e}"


def mark_done(path: Path) -> Path:
    """Move a processed command file to ``commands/done/``."""
    done = path.parent / DONE_DIR
    done.mkdir(exist_ok=True)
    target = done / path.name
    shutil.move(str(path), str(target))
    return target
