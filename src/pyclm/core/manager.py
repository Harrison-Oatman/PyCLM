"""
The Manager is the timing brain of the feedback loop: it walks the
acquisition plan and turns each timepoint into messages for the microscope,
the SLM buffer and the pattern process. The SLMBuffer holds the current DMD
pattern per experiment and answers the microscope's pattern requests.

Frame routing lives in ``router.py`` and persistence in ``writer_process.py``
(before Stage 3 both were the "microscope outbox" defined here;
``MicroscopeOutbox`` is still importable from this module).
"""

import datetime
import json
import logging
from pathlib import Path
from queue import Empty
from threading import Event
from time import sleep, time
from typing import Any

import numpy as np
from cv2 import warpAffine

from pyclm.core.pattern_process import RequestPattern

from .base_process import BaseProcess
from .datatypes import CameraPattern, EventSLMPattern, SkippedAcquisition
from .events import (
    AcquisitionEvent,
    UpdatePatternEvent,
    UpdateStagePositionEvent,
)
from .experiments import ConfigGroup, DeviceProperty, Experiment
from .grid import cut
from .messages import (
    AcquisitionEventMessage,
    CloseMessage,
    EventDoneMessage,
    Message,
    SettingsRequestMessage,
    UpdatePatternEventMessage,
    UpdatePatternParamsMessage,
    UpdatePositionEventMessage,
    UpdateZPositionMessage,
)
from .plan import AcquisitionPlan, PlannedEvent
from .queues import AllQueues
from .settings import STIMULATION, SettingChange, check_change, property_type
from .storage.events import EventLog
from .writer_process import MicroscopeOutbox, WriterProcess

logger = logging.getLogger(__name__)

__all__ = ["Manager", "MicroscopeOutbox", "SLMBuffer", "WriterProcess"]


def compose_affine(affine, roi_offset=(0.0, 0.0), binning=1) -> np.ndarray:
    """
    The camera-to-SLM affine for a pattern expressed in a camera ROI at a
    binning: ``affine`` is calibrated in full-frame unbinned pixels, so a
    pattern pixel ``p`` sits at ``binning * p + roi_offset`` in that frame.
    """
    at = np.array(affine, dtype=np.float32)
    ox, oy = roi_offset
    if ox or oy:
        at[:, 2] = at[:, 2] + at[:, :2] @ np.array([ox, oy], dtype=np.float32)
    if binning != 1:
        at[:, :2] = at[:, :2] * binning
    return at


class SLMBuffer(BaseProcess):
    """
    Holds the most recent pattern per experiment in SLM coordinates and
    answers the Manager's ``update_pattern_event`` with it (the microscope
    waits for that reply before a stimulation frame).
    """

    def __init__(self, aq: AllQueues, stop_event: Event | None = None):
        super().__init__(stop_event, name="slm buffer")

        self.from_manager = aq.manager_to_slm_buffer
        self.from_pattern = aq.pattern_to_slm
        self.to_microscope = aq.slm_to_microscope

        self.slm_patterns = {}

        self.slm_shape = None
        self.affine_transform = None

        self.initialized = False

        self.manager_done = False
        self.pattern_done = False

        self.register_queue(self.from_manager, self.handle_message)
        self.register_queue(self.from_pattern, self._handle_from_pattern)

    def initialize(
        self,
        shape: tuple[int, int],
        affine_transform: np.ndarray[Any, np.float32],
        experiment_names: list[str],
        roi=None,
        grids: dict | None = None,
    ):
        """
        :param roi: the camera ROI ``(x, y, width, height)`` in unbinned pixels
            the patterns are expressed in; the affine is calibrated in the
            full frame, so the offset is composed into it
        :param grids: ``{experiment: GridGeometry}`` for grid experiments, whose
            (stitched) patterns are cut into one DMD image per tile
        """
        self.slm_shape = shape
        self.affine_transform = np.array(affine_transform)
        self.roi_offset = (0.0, 0.0) if roi is None else (float(roi[0]), float(roi[1]))
        self.grids = dict(grids or {})

        assert affine_transform.shape == (2, 3), "Affine transform must be a 2x3 matrix"

        # Initialize patterns for each experiment
        for name in experiment_names:
            slm_pattern = np.zeros(
                self.slm_shape, dtype=np.uint8
            )  # Initialize a blank pattern
            # (pattern id, DMD image(s), camera-space uint8 pattern or None, DMD ids)
            geom = self.grids.get(name)
            if geom is None:
                self.slm_patterns[name] = (0, slm_pattern, None, None)
            else:
                n = len(geom.tiles)
                self.slm_patterns[name] = (
                    0,
                    [np.zeros(self.slm_shape, dtype=np.uint8) for _ in range(n)],
                    None,
                    [f"0:{k}" for k in range(n)],
                )

        self.initialized = True

    def pattern_to_slm(self, pattern: np.ndarray, binning=1):
        """
        This function takes a pattern and applies the stored affine transformation
        :param pattern: np array of type float scaled from 0-1, in coordinates of camera
        :param binning: the camera binning the pattern is at
        :return: at_slm_pattern: np array of type uint8, transformed to SLM coordinates
        """
        assert self.initialized, (
            "SLMBuffer must be initialized before converting patterns"
        )

        at = compose_affine(self.affine_transform, self.roi_offset, binning)

        return warpAffine(
            np.round(pattern * 255).astype(np.uint8),
            at,
            (self.slm_shape[1], self.slm_shape[0]),
        )

    def _handle_from_pattern(self, item):
        if isinstance(item, Message):
            return self.handle_message(item)
        self.handle_data(item)
        return False

    def handle_data(self, data: CameraPattern):
        logger.info("SLM buffer received a pattern")

        pattern = data.data
        pattern_id = data.pattern_id

        experiment_name = data.experiment

        camera = np.round(np.clip(np.asarray(pattern, dtype=np.float32), 0, 1) * 255)
        camera = camera.astype(np.uint8)
        geom = self.grids.get(experiment_name)
        if geom is None:
            slm_pattern = self.pattern_to_slm(pattern, data.binning)
            dmd_ids = None
        else:
            # the method returned the stitched pattern: one DMD image per tile
            expected = geom.shape(data.binning)
            if tuple(np.asarray(pattern).shape[:2]) != tuple(expected):
                logger.warning(
                    f"{experiment_name}: pattern shape {np.asarray(pattern).shape} "
                    f"is not the stitched shape {expected}; cutting what fits"
                )
            slm_pattern = [
                self.pattern_to_slm(tile, data.binning)
                for tile in cut(pattern, geom, data.binning)
            ]
            dmd_ids = [f"{pattern_id}:{k}" for k in range(len(slm_pattern))]

        # Store the pattern in the dictionary
        if experiment_name in self.slm_patterns:
            # set the current pattern and id
            self.slm_patterns[experiment_name] = (
                pattern_id,
                slm_pattern,
                camera,
                dmd_ids,
            )
        else:
            logger.warning(
                f"Experiment name '{experiment_name}' not found in SLM patterns."
            )

    def handle_message(self, msg):
        """
        Handle messages sent to the SLMBuffer from the manager (and the
        pattern process's stream close).
        :param msg: Message object
        :return: bool indicating whether to close the process
        """
        match msg.message:
            case "close":
                self.manager_done = True

            case "stream_close":
                self.pattern_done = True

            case "update_pattern_event":
                event = msg.event

                update_pattern_event_id = event.id
                experiment_name = event.experiment_name

                pattern = self.slm_patterns[experiment_name]

                data = EventSLMPattern(
                    update_pattern_event_id,
                    pattern[1],
                    pattern[0],
                    camera_pattern=pattern[2],
                    dmd_ids=pattern[3],
                )

                self.to_microscope.put(data)

            case _:
                raise ValueError(f"Unexpected message: {msg}")

        if self.manager_done and self.pattern_done:
            return True

        return False


class Manager:
    """
    Walks the acquisition plan: waits for each timepoint, then turns the plan's
    events for that timepoint into messages for the other processes.
    """

    def __init__(self, aq: AllQueues, stop_event: Event | None = None):
        self.stop_event = stop_event
        self.msgout = {
            "microscope": aq.manager_to_microscope,
            "slm_buffer": aq.manager_to_slm_buffer,
            "pattern": aq.manager_to_pattern,
        }

        self.msgin = {
            "microscope": aq.microscope_to_manager,
            "pattern": aq.pattern_to_manager,
        }

        # seconds to sleep between inbox checks while waiting for the next timepoint
        self.sleep_interval = 0.01

        self.initialized = False
        self.plan: AcquisitionPlan | None = None
        self.schedule = None
        self.experiments = None
        self.times = None
        self.positions = None

        # runtime edits (core/settings.py) and feedback from the microscope
        self.event_log: EventLog = EventLog(None)
        self.status_path: Path | None = None
        self.health = None  # optional callable returning process counters
        self.current_t = 0  # the timepoint being awaited or dispatched
        self._last_burst_t = -1  # the last timepoint whose events were emitted
        self.start_time = 0.0
        # (experiment, channel) -> {frames-table column: value} in force
        self.overrides: dict[tuple[str, str], dict] = {}
        # t -> experiment -> {"frames", "errors", "lateness_s"} from acknowledgements
        self.acks: dict[int, dict[str, dict]] = {}
        self.settings_applied = 0
        self.settings_refused = 0
        self._late_warned: set[int] = set()
        self.finished = False
        # the experiment the microscope last acknowledged (where it is now)
        self.current_experiment: str | None = None
        # commands (core/commands.py): polled from a directory at the boundary
        self.commands_dir: Path | None = None
        self.paused = False
        self._pause_started = 0.0
        self.paused_s = 0.0
        self.stopping = False
        self.pending_commands = 0
        self.commands_applied = 0
        self._last_poll = 0.0
        self.poll_interval = 0.5

    def initialize(
        self,
        plan: AcquisitionPlan,
        event_log: EventLog | None = None,
        status_path: Path | None = None,
        health=None,
        commands_dir: Path | None = None,
    ):
        self.plan = plan
        self.schedule = plan.schedule
        self.skipped = 0
        self.experiments: dict[str, Experiment] = plan.schedule.experiments
        self.positions = plan.schedule.positions
        self.times = plan.schedule.times
        if event_log is not None:
            self.event_log = event_log
        self.status_path = None if status_path is None else Path(status_path)
        self.health = health
        self.commands_dir = None if commands_dir is None else Path(commands_dir)

        self.initialized = True

    def handle_message(self, msg: Message):
        match msg.message:
            case "update_z_position":
                assert isinstance(msg, UpdateZPositionMessage)
                name = msg.experiment_name
                val = msg.new_z_position

                old = self.positions[name].z
                self.positions[name].z = val
                self.event_log.record(
                    "z_correction",
                    name,
                    None,
                    "z",
                    old,
                    val,
                    "applied",
                    "focus lock moved z",
                    self.current_t,
                    self.current_t,
                )

            case "settings_request":
                assert isinstance(msg, SettingsRequestMessage)
                self.apply_settings(msg)

            case "event_done":
                assert isinstance(msg, EventDoneMessage)
                self.handle_event_done(msg)

            case "pattern_params_result":
                for key, value in msg.applied.items():
                    self.event_log.record(
                        "pattern",
                        msg.experiment_name,
                        None,
                        key,
                        None,
                        value,
                        "applied",
                        None,
                        self.current_t,
                        self.next_timepoint(),
                        "command",
                    )
                for key, reason in msg.refused.items():
                    self.event_log.record(
                        "pattern",
                        msg.experiment_name,
                        None,
                        key,
                        None,
                        None,
                        "refused",
                        reason,
                        self.current_t,
                        self.next_timepoint(),
                        "command",
                    )

            case _:
                raise ValueError(f"Unexpected message: {msg}")

    def drain_inboxes(self) -> bool:
        """Handle every pending inbound message. Returns True if any was handled."""
        handled = False

        for inbox in self.msgin.values():
            while True:
                try:
                    msg = inbox.get_nowait()
                except Empty:
                    break

                self.handle_message(msg)
                handled = True

        return handled

    # ------------------------------------------------------ runtime edits
    def _config_for(self, name: str, channel: str | None):
        """(ImagingConfig, canonical channel name) for a channel of an experiment, or (None, channel)."""
        exp = self.experiments[name]
        stim = self.plan.stim_channel(name)
        if channel == STIMULATION or (stim is not None and channel == stim):
            return exp.stimulation, stim or STIMULATION
        return exp.channels.get(channel), channel

    def apply_settings(self, msg: SettingsRequestMessage, source: str = "pattern"):
        """
        Validate and apply a pattern method's requested changes to its own
        experiment. They take effect from ``current_t`` (the next timepoint
        whose events have not been emitted) and every one is recorded.
        """
        name = msg.experiment_name
        t_apply = self.next_timepoint()
        for change in msg.changes:
            reason = (
                None if name in self.experiments else f"unknown experiment {name!r}"
            )
            channel = change.channel
            cfg = None
            if reason is None:
                reason = check_change(change)
            if reason is None and change.kind != "position":
                cfg, channel = self._config_for(name, change.channel)
                if cfg is None:
                    reason = (
                        f"unknown channel {change.channel!r} "
                        f"(channels: {self.plan.channels(name)}, or '{STIMULATION}')"
                    )
            if reason is not None:
                self.settings_refused += 1
                logger.warning(f"{name}: refused {change.describe()}: {reason}")
                self.event_log.record(
                    change.kind,
                    name,
                    channel,
                    change.key,
                    None,
                    change.value,
                    "refused",
                    reason,
                    msg.t_requested,
                    t_apply,
                    source,
                )
                continue
            old = self._apply_change(name, cfg, channel, change)
            self.settings_applied += 1
            logger.info(f"{name}: {change.describe()} (was {old!r}) from t = {t_apply}")
            self.event_log.record(
                change.kind,
                name,
                channel,
                change.key,
                old,
                change.value,
                "applied",
                None,
                msg.t_requested,
                t_apply,
                source,
            )

    def next_timepoint(self) -> int:
        """The first timepoint whose events have not been emitted: where a change takes effect."""
        if self.current_t > self._last_burst_t:
            return self.current_t
        return self._last_burst_t + 1

    def _apply_change(self, name: str, cfg, channel: str | None, change: SettingChange):
        if change.kind == "position":
            pos = self.positions[name]
            if change.key == "pfs_offset":
                old = pos.extras.get("PFSOffset")
                pos.extras["PFSOffset"] = float(change.value)
            else:
                old = getattr(pos, change.key)
                setattr(pos, change.key, float(change.value))
            return old
        if change.kind == "exposure":
            old = cfg.exposure
            cfg.exposure = float(change.value)
            return old
        if change.kind == "config":
            old = {g.group: g.config for g in cfg.get_config_groups()}.get(change.key)
            cfg.update_config_groups([ConfigGroup(change.key, change.value)])
        else:
            old = next(
                (
                    d.value
                    for d in cfg.get_device_properties()
                    if d.device == change.device and d.property == change.prop
                ),
                None,
            )
            cfg.update_device_properties(
                [
                    DeviceProperty(
                        change.device,
                        change.prop,
                        change.value,
                        property_type(change.value),
                    )
                ]
            )
        self.overrides.setdefault((name, channel), {})[change.column] = change.value
        return old

    def construct_position_event_message(self, position, name, **kwargs):
        self.msgout["microscope"].put(
            UpdatePositionEventMessage(
                UpdateStagePositionEvent(position, name, **kwargs)
            )
        )

    # ------------------------------------------------------------ commands
    def poll_commands(self) -> bool:
        """Apply the command files waiting in ``commands_dir`` (at most every ``poll_interval`` s)."""
        if self.commands_dir is None:
            return False
        now = time()
        if now - self._last_poll < self.poll_interval:
            return False
        self._last_poll = now
        from ..commands import mark_done, pending_in

        found = pending_in(self.commands_dir)
        self.pending_commands = len(found)
        handled = False
        for path, command, problem in found:
            if command is None:
                logger.warning(f"command file {path.name} refused: {problem}")
                self.event_log.record(
                    "command",
                    None,
                    None,
                    path.name,
                    None,
                    None,
                    "refused",
                    problem,
                    self.current_t,
                    self.next_timepoint(),
                    "command",
                )
            else:
                self.apply_command(command, path.name)
            mark_done(path)
            handled = True
        self.pending_commands = 0
        return handled

    def apply_command(self, command, name: str = "") -> bool:
        """Apply one command; every outcome is an events row. Returns True if applied."""
        text = command.describe()
        t_apply = self.next_timepoint()

        def record(status, detail=None, old=None, new=None):
            self.event_log.record(
                "command",
                command.experiment,
                command.channel,
                command.command,
                old,
                new if new is not None else text,
                status,
                detail,
                self.current_t,
                t_apply,
                "command",
            )

        kind = command.command
        if kind == "pause":
            if not self.paused:
                self.paused = True
                self._pause_started = time()
            logger.warning("run paused by command")
            record("applied")
        elif kind == "resume":
            if not self.paused:
                record("refused", "not paused")
                return False
            shift = time() - self._pause_started
            self.start_time += shift
            self.paused_s += shift
            self.paused = False
            logger.warning(f"run resumed after {shift:.1f}s")
            record("applied", f"paused {shift:.1f} s")
        elif kind == "stop_run":
            self.stopping = True
            logger.warning("run will stop after this timepoint (command)")
            record("applied")
        elif kind == "stop_experiment":
            if not self.plan.stop_experiment(command.experiment):
                record(
                    "refused",
                    f"unknown or already stopped experiment {command.experiment!r}",
                )
                return False
            logger.warning(f"{command.experiment} stopped by command")
            record("applied")
        elif kind == "set_pattern":
            if command.experiment not in self.experiments:
                record("refused", f"unknown experiment {command.experiment!r}")
                return False
            self.msgout["pattern"].put(
                UpdatePatternParamsMessage(command.experiment, command.parameters)
            )
            record("applied", "sent to the pattern method; see the pattern rows")
        else:
            before = self.settings_refused
            self.apply_settings(
                SettingsRequestMessage(
                    command.experiment, self.current_t, command.to_changes()
                ),
                source="command",
            )
            refused = self.settings_refused - before
            if refused:
                record("refused", f"{refused} change(s) refused; see the rows above")
                return False
            record("applied")
        self.commands_applied += 1
        return True

    # -------------------------------------------------- acknowledgements
    def handle_event_done(self, msg: EventDoneMessage):
        rec = self.acks.setdefault(msg.t_index, {}).setdefault(
            msg.experiment_name,
            {"frames": 0, "errors": 0, "lateness_s": 0.0, "skipped": 0},
        )
        if getattr(msg, "skipped", False):
            rec["skipped"] = rec.get("skipped", 0) + 1
            self.skipped += 1
            self.event_log.record(
                "position_skipped",
                msg.experiment_name,
                msg.channel,
                None,
                None,
                None,
                "skipped",
                "only the stimulation frame was due and the pattern was blank",
                msg.t_index,
                msg.t_index,
            )
            return
        rec["frames"] += 1
        self.current_experiment = msg.experiment_name
        if msg.error:
            rec["errors"] += 1
            self.event_log.record(
                "acquisition_error",
                msg.experiment_name,
                msg.channel,
                None,
                None,
                None,
                "failed",
                msg.error,
                msg.t_index,
                msg.t_index,
            )
            return
        late = msg.lateness_s or 0.0
        rec["lateness_s"] = max(rec["lateness_s"], late)
        if late > self.plan.interval_s and msg.t_index not in self._late_warned:
            self._late_warned.add(msg.t_index)
            logger.warning(
                f"t = {msg.t_index}: {msg.experiment_name}/{msg.channel} completed "
                f"{late:.1f}s late (interval {self.plan.interval_s:.1f}s)"
            )
            self.event_log.record(
                "late",
                msg.experiment_name,
                msg.channel,
                None,
                None,
                f"{late:.3f}",
                "warning",
                "completed more than one interval late",
                msg.t_index,
                msg.t_index,
            )

    def status(self, done: bool = False) -> dict:
        """What the run looks like now: progress, per-experiment lateness and errors, process health."""
        experiments = {}
        for name in self.plan.experiments:
            seen = [(t, recs[name]) for t, recs in self.acks.items() if name in recs]
            errors = sum(r["errors"] for _, r in seen)
            skipped = sum(r.get("skipped", 0) for _, r in seen)
            if seen:
                t_last, last = max(seen, key=lambda item: item[0])
                experiments[name] = {
                    "last_t": t_last,
                    "lateness_s": round(last["lateness_s"], 3),
                    "errors": errors,
                    "skipped": skipped,
                }
            else:
                experiments[name] = {
                    "last_t": None,
                    "lateness_s": None,
                    "errors": 0,
                    "skipped": 0,
                }
        return {
            "t": self.current_t,
            "timepoints": self.plan.timepoints,
            "done": done,
            "current_experiment": self.current_experiment,
            "wall_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time() - self.start_time, 1) if self.start_time else 0.0,
            "settings_applied": self.settings_applied,
            "settings_refused": self.settings_refused,
            "paused": self.paused,
            "paused_s": round(
                self.paused_s + (time() - self._pause_started if self.paused else 0.0),
                1,
            ),
            "stopping": self.stopping,
            "pending_commands": self.pending_commands,
            "commands_applied": self.commands_applied,
            "skipped": self.skipped,
            "experiments": experiments,
            "health": self.health() if self.health is not None else {},
        }

    def write_status(self, done: bool = False):
        if self.status_path is None:
            return
        try:
            tmp = self.status_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.status(done), indent=1, default=str))
            tmp.replace(self.status_path)
        except Exception as e:
            logger.error(f"failed to write {self.status_path}: {e}")

    def finish(self):
        """Absorb the last acknowledgements, write the final status, close the events table."""
        if self.finished or not self.initialized:
            return
        self.finished = True
        self.drain_inboxes()
        self.write_status(done=True)
        self.event_log.close()

    def dispatch(self, ev: PlannedEvent, start_time: float):
        """Turn one planned event into the message(s) the other processes expect."""
        name = ev.experiment
        scheduled_time = start_time + ev.scheduled_offset_s
        since_start = ev.scheduled_offset_s

        match ev.kind:
            case "request_pattern":
                self.msgout["pattern"].put(
                    RequestPattern(
                        ev.t, since_start, name, self.plan.requirements.get(name, [])
                    )
                )

            case "position":
                self.construct_position_event_message(
                    self.positions[name],
                    name,
                    index=ev.index,
                    skippable_if_blank=ev.skippable,
                )

            case "update_pattern":
                cfg = self.plan.imaging_config(name, ev.channel)
                upmsg = UpdatePatternEventMessage(
                    UpdatePatternEvent(
                        name,
                        cfg.get_config_groups(),
                        cfg.get_device_properties(),
                        index=ev.index,
                        skippable_if_blank=ev.skippable,
                    )
                )
                self.msgout["slm_buffer"].put(upmsg)
                self.msgout["microscope"].put(upmsg)

            case "acquire":
                cfg = self.plan.imaging_config(name, ev.channel)
                event = AcquisitionEvent(
                    name,
                    self.positions[name],
                    cfg.channel_id,
                    index=ev.index,
                    scheduled_time=scheduled_time,
                    scheduled_time_since_start=since_start,
                    exposure_time_ms=cfg.exposure,
                    needs_slm=ev.is_stim,
                    config_groups=cfg.get_config_groups(),
                    devices=cfg.get_device_properties(),
                    save_output=ev.save,
                    binning=cfg.binning,
                )
                event.overrides = dict(self.overrides.get((name, ev.channel), {}))
                event.skippable_if_blank = ev.skippable
                self.msgout["microscope"].put(AcquisitionEventMessage(event))

            case _:
                raise ValueError(f"unknown planned event kind {ev.kind!r}")

    def process(self):
        assert self.initialized, (
            "manager must be initialized with an acquisition plan to start"
        )

        plan = self.plan
        setup = plan.setup_s
        self.start_time = time() + setup

        # time iter loop
        for t in range(plan.timepoints):
            self.current_t = t
            # operator-facing progress line (the console log handler only shows warnings)
            print(f"t = {t}: {(time() - self.start_time) / 60: 0.1f} minutes")

            # wait until the preparatory phase; settings requests, acknowledgements
            # and command files are handled here, so they apply at this boundary
            while True:
                if self.stop_event and self.stop_event.is_set():
                    logger.info("force stopping manager process")
                    return
                due = (time() - self.start_time) >= plan.time_offset_s(t) - setup
                if due and not self.paused:
                    break
                handled = self.drain_inboxes()
                handled = self.poll_commands() or handled
                if not handled:
                    sleep(self.sleep_interval)

            if self.stopping:
                logger.info(f"stopping before t = {t} (stop_run command)")
                self.write_status()
                break

            self.write_status()
            for ev in plan.events_at(t):
                self.dispatch(ev, self.start_time)
            self._last_burst_t = t

        print("DONE")
        logger.info("Manager finished the schedule; sending close to all processes")

        for box in self.msgout:
            self.msgout[box].put(CloseMessage())
