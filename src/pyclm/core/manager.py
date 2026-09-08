"""
The Manager is the timing brain of the feedback loop: it walks the
acquisition plan and turns each timepoint into messages for the microscope,
the SLM buffer and the pattern process. The SLMBuffer holds the current DMD
pattern per experiment and answers the microscope's pattern requests.

Frame routing lives in ``router.py`` and persistence in ``writer_process.py``
(before Stage 3 both were the "microscope outbox" defined here;
``MicroscopeOutbox`` is still importable from this module).
"""

import logging
from queue import Empty
from threading import Event
from time import sleep, time
from typing import Any

import numpy as np
from cv2 import warpAffine

from pyclm.core.pattern_process import RequestPattern

from .base_process import BaseProcess
from .datatypes import CameraPattern, EventSLMPattern
from .events import (
    AcquisitionEvent,
    UpdatePatternEvent,
    UpdateStagePositionEvent,
)
from .experiments import Experiment
from .messages import (
    AcquisitionEventMessage,
    CloseMessage,
    Message,
    UpdatePatternEventMessage,
    UpdatePositionEventMessage,
    UpdateZPositionMessage,
)
from .plan import AcquisitionPlan, PlannedEvent
from .queues import AllQueues
from .writer_process import MicroscopeOutbox, WriterProcess

logger = logging.getLogger(__name__)

__all__ = ["Manager", "MicroscopeOutbox", "SLMBuffer", "WriterProcess"]


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
    ):
        self.slm_shape = shape
        self.affine_transform = np.array(affine_transform)

        assert affine_transform.shape == (2, 3), "Affine transform must be a 2x3 matrix"

        # Initialize patterns for each experiment
        for name in experiment_names:
            slm_pattern = np.zeros(
                self.slm_shape, dtype=np.uint8
            )  # Initialize a blank pattern
            self.slm_patterns[name] = (
                0,
                slm_pattern,
            )  # Store the pattern in the dictionary

        self.initialized = True

    def pattern_to_slm(self, pattern: np.ndarray, slm_coords=False, binning=1):
        """
        This function takes a pattern and applies the stored affine transformation
        :param pattern: np array of type float scaled from 0-1, in coordinates of camera
        :param slm_coords: bool whether pattern is already in slm coordinate space
        :return: at_slm_pattern: np array of type uint8, transformed to SLM coordinates
        """
        assert self.initialized, (
            "SLMBuffer must be initialized before converting patterns"
        )

        if slm_coords:
            return np.round(pattern).astype(np.uint8)

        at = np.copy(self.affine_transform)
        if binning != 1:
            at[:, :2] = at[:, :2] * binning

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

        slm_pattern = self.pattern_to_slm(pattern, data.slm_coords, data.binning)

        # Store the pattern in the dictionary
        if experiment_name in self.slm_patterns:
            # set the current pattern and id
            self.slm_patterns[experiment_name] = (pattern_id, slm_pattern)
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

                data = EventSLMPattern(update_pattern_event_id, pattern[1], pattern[0])

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
        }

        # seconds to sleep between inbox checks while waiting for the next timepoint
        self.sleep_interval = 0.01

        self.initialized = False
        self.plan: AcquisitionPlan | None = None
        self.schedule = None
        self.experiments = None
        self.times = None
        self.positions = None

    def initialize(self, plan: AcquisitionPlan):
        self.plan = plan
        self.schedule = plan.schedule
        self.experiments: dict[str, Experiment] = plan.schedule.experiments
        self.positions = plan.schedule.positions
        self.times = plan.schedule.times

        self.initialized = True

    def handle_message(self, msg: Message):
        match msg.message:
            case "update_z_position":
                assert isinstance(msg, UpdateZPositionMessage)
                name = msg.experiment_name
                val = msg.new_z_position

                self.positions[name].z = val

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

    def construct_position_event_message(self, position, name):
        self.msgout["microscope"].put(
            UpdatePositionEventMessage(UpdateStagePositionEvent(position, name))
        )

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
                self.construct_position_event_message(self.positions[name], name)

            case "update_pattern":
                cfg = self.plan.imaging_config(name, ev.channel)
                upmsg = UpdatePatternEventMessage(
                    UpdatePatternEvent(
                        name, cfg.get_config_groups(), cfg.get_device_properties()
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
                self.msgout["microscope"].put(AcquisitionEventMessage(event))

            case _:
                raise ValueError(f"unknown planned event kind {ev.kind!r}")

    def process(self):
        assert self.initialized, (
            "manager must be initialized with an acquisition plan to start"
        )

        plan = self.plan
        setup = plan.setup_s
        start_time = time() + setup

        # time iter loop
        for t in range(plan.timepoints):
            # operator-facing progress line (the console log handler only shows warnings)
            print(f"t = {t}: {(time() - start_time) / 60: 0.1f} minutes")

            # wait until preparatory phase
            # todo: check if we are behind schedule
            while (time() - start_time) < plan.time_offset_s(t) - setup:
                if self.stop_event and self.stop_event.is_set():
                    logger.info("force stopping manager process")
                    return

                if not self.drain_inboxes():
                    sleep(self.sleep_interval)

            for ev in plan.events_at(t):
                self.dispatch(ev, start_time)

        print("DONE")
        logger.info("Manager finished the schedule; sending close to all processes")

        for box in self.msgout:
            self.msgout[box].put(CloseMessage())
