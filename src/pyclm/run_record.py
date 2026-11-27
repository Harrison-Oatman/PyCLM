"""
``run_record.json``: a short summary of one run, written when it ends.

Everything about a run is already in its directory, but spread over files
that need PyCLM (or the data) to read. The record is the one small file a
person, a script or an agent can read to learn what was run, where, with
which PyCLM and configuration, and how it went, without opening the data:

- when: start and end, with the UTC offset, and the duration;
- where: host, user, the command line, dry or hardware;
- PyCLM: version, and the git commit when it runs from a git checkout;
- configuration: the paths and SHA-256 of ``pyclm_config.toml`` and of the
  Micro-Manager ``.cfg``;
- per experiment: the pattern method and its arguments, the segmentation
  and tracking methods, and the last timepoint acquired, with its errors;
- the outcome (``finished``, ``stopped``, ``crashed``), the error message
  of a crash, and the error, dropped-frame and undeliverable counts of
  ``status.json``;
- flags raised by ``pyclm watch`` up to the end of the run (by kind);
- the outputs, with their sizes.

The record is written by ``pyclm run`` itself, so it exists whether or not
the run was started by a person, a script or an agent.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RECORD_NAME = "run_record.json"
FORMAT_VERSION = 1


def now() -> datetime.datetime:
    """The local time with its UTC offset."""
    return datetime.datetime.now().astimezone()


def sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, TypeError):
        return None


def pyclm_version() -> dict:
    """The installed version and, from a git checkout, the commit."""
    out: dict = {"version": None, "commit": None, "dirty": None}
    try:
        from importlib.metadata import version

        out["version"] = version("closed-loop-microscopy")
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            out["commit"] = commit.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=here,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if dirty.returncode == 0:
                out["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def size_bytes(path: Path) -> int | None:
    path = Path(path)
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return None
    return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def watch_flags(directory: Path, until: datetime.datetime | None = None) -> dict:
    """Flags ``pyclm watch`` raised (by kind: count of checks that showed one)."""
    log = Path(directory) / "watch" / "flags_log.jsonl"
    kinds: dict[str, int] = {}
    try:
        lines = log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return kinds
    limit = until.replace(tzinfo=None) if until is not None else None
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if limit is not None:
            try:
                if datetime.datetime.fromisoformat(entry["checked"]) > limit:
                    continue
            except (KeyError, ValueError):
                pass
        for flag in entry.get("flags", []):
            kind = flag.get("kind", "?")
            kinds[kind] = kinds.get(kind, 0) + 1
    return kinds


def describe_experiments(schedule, status: dict | None) -> dict:
    """Per experiment: its TOML, methods and arguments, and how far it got."""
    progress = (status or {}).get("experiments", {})
    out = {}
    for name, exp in schedule.experiments.items():
        pattern = exp.pattern
        segs = {
            key: seg.method_name
            for key, seg in exp.segmentations.items()
            if seg.method_name != "none"
        }
        entry = {
            "toml": f"{str(name).split('.')[0]}.toml",
            "pattern": {
                "method": pattern.method_name,
                "kwargs": dict(pattern.kwargs),
            },
            "segmentation": segs,
            "tracking": exp.tracking.method_name
            if exp.tracking.method_name != "none"
            else None,
            "channels": list(exp.channels),
        }
        if exp.slm is not None:
            entry["slm"] = exp.slm
        run = progress.get(name)
        if run is not None:
            entry["last_t"] = run.get("last_t")
            entry["errors"] = run.get("errors")
            entry["skipped"] = run.get("skipped")
        out[name] = entry
    return out


def outcome_of(status: dict | None, error: BaseException | None) -> str:
    if error is not None:
        return "crashed"
    if not status:
        return "unknown"
    if status.get("stopping"):
        return "stopped"
    if status.get("done"):
        return "finished"
    return "unknown"


def build_record(
    directory,
    *,
    started: datetime.datetime,
    ended: datetime.datetime | None = None,
    dry: bool = False,
    config_path=None,
    mm_config_path=None,
    schedule=None,
    outputs: dict | None = None,
    error: BaseException | None = None,
    argv: list[str] | None = None,
) -> dict:
    directory = Path(directory)
    ended = now() if ended is None else ended
    status = _read_json(directory / "status.json")
    health = (status or {}).get("health", {})
    errors = health.get("errors", {})
    record = {
        "format_version": FORMAT_VERSION,
        "directory": str(directory.resolve()),
        "started": started.isoformat(timespec="seconds"),
        "ended": ended.isoformat(timespec="seconds"),
        "duration_s": round((ended - started).total_seconds(), 1),
        "outcome": outcome_of(status, error),
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "dry": bool(dry),
        "host": platform.node(),
        "user": _user(),
        "platform": sys.platform,
        "command": list(sys.argv if argv is None else argv),
        "pyclm": pyclm_version(),
        "config": {
            "pyclm_config": None if config_path is None else str(config_path),
            "pyclm_config_sha256": sha256(config_path),
            "mm_config": None if mm_config_path is None else str(mm_config_path),
            "mm_config_sha256": sha256(mm_config_path),
        },
        "timepoints": {
            "planned": (status or {}).get("timepoints")
            or (schedule.times.count if schedule is not None else None),
            "last_acquired": (status or {}).get("acquired_t"),
        },
        "errors": {
            "by_process": errors,
            "total": sum(v for v in errors.values() if isinstance(v, int | float)),
            "dropped_frames": health.get("dropped_frames"),
            "undeliverable": health.get("undeliverable"),
        },
        "watch_flags": watch_flags(directory, until=ended),
        "experiments": {}
        if schedule is None
        else describe_experiments(schedule, status),
        "outputs": {
            name: {"path": str(path), "bytes": size_bytes(path)}
            for name, path in (outputs or {}).items()
        },
    }
    return record


def write_run_record(directory, **kwargs) -> Path | None:
    """Write ``run_record.json`` in ``directory``; failures are logged, never raised."""
    path = Path(directory) / RECORD_NAME
    try:
        record = build_record(directory, **kwargs)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        tmp.replace(path)
        return path
    except Exception as e:
        logger.error(f"failed to write {path}: {e}", exc_info=True)
        return None


def _user() -> str:
    from .lock import _user as person

    return person()
