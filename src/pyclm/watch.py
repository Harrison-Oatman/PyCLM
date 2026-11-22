"""
``pyclm watch``: follow a run from its own files and flag what needs a person.

::

    pyclm watch <dir>                          # every 60 s until the run is done
    pyclm watch <dir> --once                   # one check; exit code 1 if anything is flagged
    pyclm watch <dir> --images                 # also brightness, saturation and focus
    pyclm watch <dir> --on-flag "notify.bat"   # run a command when something new is flagged
    pyclm watch <dir> --then "next_step.bat"   # run a command once the run is done

It reads only what a run writes anyway: ``status.json``, ``log.log``,
``detached.json`` (from ``pyclm run --detached``) and, with ``--images``, the
newest frames through :mod:`pyclm.io`. It never writes into the experiment
directory except its own ``watch/`` folder, and never into ``commands/``: it
cannot change or stop a run.

Flags (kind, then what raises it):

``no_status``        no ``status.json`` yet
``stale``            ``status.json`` not rewritten for ``--stale-min`` minutes while the run is not done
``stopped``          the process ``pyclm run --detached`` started has exited, and the run is not done
``late``             an experiment's last acquisition was more than ``--late-s`` late (default: one interval)
``acquisition_error`` an experiment's count of failed acquisitions rose
``process_error``    a process's error count in ``status.json`` health rose
``dropped_frames``   the writer dropped frames, or the router could not deliver some
``log_error``        new ERROR or CRITICAL lines, or a traceback, in ``log.log``
``brightness``       (``--images``) a channel's newest frame is far darker or brighter than its first
``saturation``       (``--images``) more than 1 % of a frame's pixels sit at the camera's maximum
``focus``            (``--images``) a channel's newest frame is much less sharp than its first

Each check writes ``watch/flags.json`` (the latest check) and appends to
``watch/flags_log.jsonl``. ``--on-flag`` runs a command, with the path of
``flags.json`` as its last argument, when a flag appears that it has not been
run for in the last ``--quiet-min`` minutes; how a person is told (email, a
phone push, a chat message) is up to that command. ``--then`` waits until the
run says it is done and its files have stopped changing, then runs its command
and exits with that command's exit code.

The thresholds are starting points, not measured limits; change them with the
options or :class:`WatchSettings`.
"""

from __future__ import annotations

import datetime
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

LOG_ERROR = re.compile(r"\b(ERROR|CRITICAL)\b|^Traceback \(most recent call last\)")
SETTLED_FILES = ("status.json", "log.log", "frames.parquet", "events.parquet")


@dataclass
class WatchSettings:
    every_s: float = 60.0
    images: bool = False
    late_s: float | None = None  # None: one plan interval
    stale_min: float = 10.0
    on_flag: str | None = None
    quiet_min: float = 30.0
    then: str | None = None
    out: Path | None = None
    settle_s: float = 30.0  # files unchanged this long before --then runs
    # --images thresholds, as ratios to the channel's first frame
    brightness_low: float = 0.5
    brightness_high: float = 2.0
    saturated_fraction: float = 0.01
    focus_low: float = 0.5


@dataclass
class Flag:
    kind: str
    where: str
    message: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.where}"


@dataclass
class _State:
    log_offset: int = 0
    process_errors: dict = field(default_factory=dict)
    experiment_errors: dict = field(default_factory=dict)
    dropped: int = 0
    undeliverable: int = 0
    first_check: bool = True
    image_refs: dict = field(
        default_factory=dict
    )  # (output, group, channel) -> (median, sharpness)
    notified: dict = field(default_factory=dict)  # flag key -> time last notified
    settled_since: float | None = None
    last_mtimes: tuple | None = None


def read_status(directory: Path) -> tuple[dict | None, float | None]:
    """``status.json`` and its modification time, or (None, None); retried once if caught half-written."""
    path = Path(directory) / "status.json"
    for _ in range(2):
        try:
            return json.loads(path.read_text()), path.stat().st_mtime
        except FileNotFoundError:
            return None, None
        except (OSError, ValueError):
            time.sleep(0.2)
    return None, None


def plan_interval_s(directory: Path) -> float | None:
    """The schedule's ``interval_seconds``, or None."""
    try:
        from .directories import find_schedule
        from .schema import ScheduleConfig

        path = find_schedule(Path(directory))
        if path is None:
            return None
        return float(ScheduleConfig.from_file(path).timing.interval_seconds)
    except Exception:
        return None


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian over the mean squared: a focus measure that ignores overall brightness."""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 3:
        return 0.0
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    mean = a.mean()
    return float(lap.var() / (mean * mean)) if mean > 0 else 0.0


class Watcher:
    """One watch of one experiment directory; :meth:`check` returns the flags now."""

    def __init__(self, directory: Path, settings: WatchSettings | None = None):
        self.directory = Path(directory)
        self.settings = settings or WatchSettings()
        self.out = self.settings.out or (self.directory / "watch")
        self.state = _State()
        self.interval_s = plan_interval_s(self.directory)
        self.status: dict | None = None

    # ------------------------------------------------------------ checks
    def check(self, now: float | None = None) -> list[Flag]:
        now = time.time() if now is None else now
        s = self.settings
        flags: list[Flag] = []
        status, mtime = read_status(self.directory)
        self.status = status
        done = bool(status and status.get("done"))

        if status is None:
            flags.append(Flag("no_status", "", "no status.json yet"))
        else:
            age_min = (now - mtime) / 60.0
            if not done and age_min > s.stale_min:
                flags.append(
                    Flag(
                        "stale",
                        "",
                        f"status.json last written {age_min:.0f} min ago; the run may have stopped",
                    )
                )
            flags += self._status_flags(status)

        if not done:
            flags += self._process_flags()
        flags += self._log_flags()
        if s.images:
            flags += self._image_flags()
        self.state.first_check = False
        return flags

    def _status_flags(self, status: dict) -> list[Flag]:
        s, st = self.settings, self.state
        flags = []
        late_s = s.late_s if s.late_s is not None else self.interval_s
        for name, info in (status.get("experiments") or {}).items():
            late = info.get("lateness_s")
            if late_s is not None and late is not None and late > late_s:
                flags.append(
                    Flag(
                        "late",
                        name,
                        f"last acquisition {late:.0f} s late (limit {late_s:.0f} s)",
                    )
                )
            errors = info.get("errors") or 0
            before = st.experiment_errors.get(name, 0)
            if errors > before:
                flags.append(
                    Flag(
                        "acquisition_error",
                        name,
                        f"{errors - before} new failed acquisition(s), {errors} in all",
                    )
                )
            st.experiment_errors[name] = errors
        health = status.get("health") or {}
        for process, count in (health.get("errors") or {}).items():
            before = st.process_errors.get(process, 0)
            if count and count > before:
                flags.append(
                    Flag(
                        "process_error",
                        process,
                        f"{count - before} new error(s) in {process}, {count} in all",
                    )
                )
            st.process_errors[process] = count or 0
        dropped = health.get("dropped_frames") or 0
        undeliverable = health.get("undeliverable") or 0
        if dropped > st.dropped or undeliverable > st.undeliverable:
            flags.append(
                Flag(
                    "dropped_frames",
                    "",
                    f"{dropped} dropped and {undeliverable} undeliverable frame(s) so far",
                )
            )
        st.dropped, st.undeliverable = dropped, undeliverable
        return flags

    def _process_flags(self) -> list[Flag]:
        from .detach import process_alive, read_record

        record = read_record(self.directory)
        if not record:
            return []
        alive = process_alive(record.get("pid"))
        if alive is False:
            return [
                Flag(
                    "stopped",
                    "",
                    f"the run's process (pid {record.get('pid')}) has exited before the "
                    f"run finished; see {record.get('log', 'detached.log')}",
                )
            ]
        return []

    def _log_flags(self) -> list[Flag]:
        path = self.directory / "log.log"
        try:
            size = path.stat().st_size
        except OSError:
            return []
        st = self.state
        if size < st.log_offset:  # a new run started over the log
            st.log_offset = 0
        try:
            with open(path, "rb") as f:
                f.seek(st.log_offset)
                chunk = f.read()
        except OSError:
            return []
        # only whole lines; a half-written last line waits for the next check
        end = chunk.rfind(b"\n") + 1
        st.log_offset += end
        lines = chunk[:end].decode("utf-8", errors="replace").splitlines()
        bad = [line.strip() for line in lines if LOG_ERROR.search(line)]
        if not bad:
            return []
        shown = "; ".join(bad[:3]) + (
            f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""
        )
        return [Flag("log_error", "log.log", f"{len(bad)} new error line(s): {shown}")]

    def _image_flags(self) -> list[Flag]:
        from . import io as pyclm_io

        s, st = self.settings, self.state
        flags = []
        for path in pyclm_io.find_experiments(self.directory):
            try:
                exp = pyclm_io.open(path)
            except Exception:
                continue
            try:
                for gname, group in exp.groups.items():
                    slots = group.acquired()
                    if not slots:
                        continue
                    for channel in group.channels:
                        key = (path.name, gname, channel)
                        where = f"{exp.name}/{channel}"
                        if key not in st.image_refs:
                            first = group.frame(slots[0], channel)
                            if first is None:
                                continue
                            st.image_refs[key] = (
                                float(np.median(first)),
                                sharpness(first),
                            )
                        frame = group.frame(slots[-1], channel)
                        if frame is None or len(slots) < 2:
                            continue
                        ref_median, ref_sharp = st.image_refs[key]
                        median = float(np.median(frame))
                        if ref_median > 0:
                            ratio = median / ref_median
                            if ratio < s.brightness_low or ratio > s.brightness_high:
                                flags.append(
                                    Flag(
                                        "brightness",
                                        where,
                                        f"median {median:.0f}, {ratio:.2f} times the first frame's",
                                    )
                                )
                        if np.issubdtype(frame.dtype, np.integer):
                            top = np.iinfo(frame.dtype).max
                            frac = float(np.mean(frame >= top))
                            if frac > s.saturated_fraction:
                                flags.append(
                                    Flag(
                                        "saturation",
                                        where,
                                        f"{100 * frac:.1f} % of pixels at {top}",
                                    )
                                )
                        if ref_sharp > 0:
                            ratio = sharpness(frame) / ref_sharp
                            if ratio < s.focus_low:
                                flags.append(
                                    Flag(
                                        "focus",
                                        where,
                                        f"sharpness {ratio:.2f} times the first frame's",
                                    )
                                )
            finally:
                exp.close()
        return flags

    # ------------------------------------------------------------ output
    def record(self, flags: list[Flag], now: float | None = None) -> Path:
        """Write ``flags.json`` and append to ``flags_log.jsonl``; returns the flags file."""
        now = time.time() if now is None else now
        self.out.mkdir(parents=True, exist_ok=True)
        st = self.status or {}
        entry = {
            "checked": datetime.datetime.fromtimestamp(now).isoformat(
                timespec="seconds"
            ),
            "directory": str(self.directory),
            "t": st.get("t"),
            "acquired_t": st.get("acquired_t"),
            "timepoints": st.get("timepoints"),
            "done": bool(st.get("done")),
            "flags": [asdict(f) for f in flags],
        }
        path = self.out / "flags.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry, indent=1))
        tmp.replace(path)
        with open(self.out / "flags_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return path

    def to_notify(self, flags: list[Flag], now: float | None = None) -> list[Flag]:
        """The flags ``--on-flag`` has not been run for within ``quiet_min``; marks them notified."""
        now = time.time() if now is None else now
        quiet = self.settings.quiet_min * 60.0
        fresh = [
            f for f in flags if now - self.state.notified.get(f.key, -1e18) >= quiet
        ]
        for f in fresh:
            self.state.notified[f.key] = now
        return fresh

    def settled(self, now: float | None = None) -> bool:
        """The run says done and its files have not changed for ``settle_s``."""
        now = time.time() if now is None else now
        if not (self.status and self.status.get("done")):
            self.state.settled_since = None
            return False
        mtimes = []
        for name in SETTLED_FILES:
            try:
                mtimes.append((self.directory / name).stat().st_mtime)
            except OSError:
                mtimes.append(None)
        mtimes = tuple(mtimes)
        if mtimes != self.state.last_mtimes:
            self.state.last_mtimes = mtimes
            self.state.settled_since = now
        return now - (self.state.settled_since or now) >= self.settings.settle_s


def run_hook(command: str, *extra: str) -> int:
    """Run a user's command through the shell, with ``extra`` arguments appended (quoted)."""
    if sys.platform == "win32":
        line = command + "".join(f' "{e}"' for e in extra)
    else:
        line = command + "".join(f" {shlex.quote(e)}" for e in extra)
    try:
        return subprocess.run(line, shell=True, timeout=600).returncode
    except subprocess.TimeoutExpired:
        print(f"watch: '{command}' did not finish within 10 min")
        return 124


def _line(flags: list[Flag], status: dict | None) -> str:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    where = ""
    if status:
        where = f" t {status.get('acquired_t')}/{status.get('timepoints')}"
        if status.get("done"):
            where += " done"
    if not flags:
        return f"{now}{where}: ok"
    return f"{now}{where}: " + "; ".join(
        f"{f.kind}{' ' + f.where if f.where else ''}: {f.message}" for f in flags
    )


def watch(
    directory: Path, settings: WatchSettings | None = None, once: bool = False
) -> int:
    """
    Watch ``directory`` until its run is done (or once). Returns the exit code:
    with ``once``, 1 if anything was flagged; with ``then``, that command's code;
    otherwise 0 when the run is done.
    """
    directory = Path(directory)
    if not directory.is_dir():
        print(f"error: {directory} is not a directory")
        return 2
    watcher = Watcher(directory, settings)
    s = watcher.settings
    while True:
        flags = watcher.check()
        path = watcher.record(flags)
        print(_line(flags, watcher.status), flush=True)
        if s.on_flag and flags:
            fresh = watcher.to_notify(flags)
            if fresh:
                run_hook(s.on_flag, str(path))
        if once:
            return 1 if flags else 0
        if watcher.status and watcher.status.get("done"):
            if not s.then:
                return 0
            if watcher.settled():
                print(f"run done and settled; running: {s.then}", flush=True)
                return run_hook(s.then)
        time.sleep(
            s.every_s if not (watcher.status or {}).get("done") else min(s.every_s, 5.0)
        )
