"""
The controller is the brain of the feedback loop.

It is responsible for
- managing timing
- passing messages between processes
- scheduling microscope events
"""

import json
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from queue import Empty
from threading import Event
from time import sleep, time
from typing import Any

import numpy as np
from cv2 import warpAffine
from h5py import File

from pyclm.core.pattern_process import RequestPattern

from .core_interface import MicroscopeCoreInterface
from .datatypes import (
    AcquisitionData,
    CameraPattern,
    EventSLMPattern,
    GenericData,
    SegmentationData,
    StimulationData,
)
from .events import (
    AcquisitionEvent,
    UpdatePatternEvent,
    UpdateStagePositionEvent,
    storage_group,
)
from .experiments import (
    Experiment,
    ExperimentSchedule,
    ImagingConfig,
    TimeCourse,
)
from .messages import (
    AcquisitionEventMessage,
    CloseMessage,
    Message,
    StreamCloseMessage,
    UpdatePatternEventMessage,
    UpdatePositionEventMessage,
    UpdateZPositionMessage,
)
from .patterns import AcquiredImageRequest
from .plan import AcquisitionPlan, PlannedEvent
from .queues import AllQueues
from .storage import FrameWriter, HDF5WriterV1

logger = logging.getLogger(__name__)


from .base_process import BaseProcess


class DataPassingProcess(BaseProcess, metaclass=ABCMeta):
    def __init__(self, aq: AllQueues, stop_event: Event | None = None):
        super().__init__(stop_event, name="data passing process")
        self.all_queues = aq

        # Subclasses should set these or register queues manually
        self.from_manager = None
        self.data_in = None

    def initialize_queues(self):
        # Helper to register standard queues if subclasses set attributes
        if self.from_manager:
            self.register_queue(self.from_manager, self.handle_message_wrapper)

        if self.data_in:
            for q in self.data_in:
                self.register_queue(q, self.handle_data_wrapper)

    def handle_message_wrapper(self, msg):
        """Wrapper to handle return value logic expected by BaseProcess"""
        if isinstance(msg, Message):
            # BaseProcess expects True to stop
            return self.handle_message(msg)
        return False

    def handle_data_wrapper(self, data):
        """Wrapper to handle data or message in data channel"""
        if isinstance(data, Message):
            logger.debug(
                f"{self.name} received message on data channel: {data.message}"
            )
            return self.handle_message(data)

        assert isinstance(data, GenericData), (
            f"Unexpected data type: {type(data)}, expected subtype of GenericData"
        )
        self.handle_data(data)
        return False

    @abstractmethod
    def handle_data(self, data):
        pass

    def handle_message(self, msg):
        match msg.message:
            case "close":
                return True

            case _:
                raise ValueError(f"Unexpected message: {msg}")


class MicroscopeOutbox(DataPassingProcess):
    """
    Receives every frame the microscope produces, hands it to the configured
    FrameWriter for persistence, and routes it on to segmentation and pattern
    generation according to the event's flags.
    """

    def __init__(
        self,
        aq: AllQueues,
        base_path: Path | None = None,
        stop_event: Event | None = None,
        writer: FrameWriter | None = None,
    ):
        super().__init__(aq, stop_event)
        self.name = "microscope outbox"

        if base_path is None:
            base_path = Path().cwd()

        self.from_manager = aq.manager_to_outbox
        self.data_in = [aq.acquisition_outbox, aq.seg_to_outbox]

        self.manager_done = False
        self.stream_count = 0

        self.seg_queue = aq.outbox_to_seg
        self.pattern_queue = aq.outbox_to_pattern

        self.base_path = base_path
        self.writer: FrameWriter = writer if writer is not None else HDF5WriterV1()
        self.plan: AcquisitionPlan | None = None

        self.initialize_queues()

    @property
    def open_files(self) -> dict:
        """Open HDF5 handles when the HDF5 writer is in use (empty otherwise)."""
        return getattr(self.writer, "open_files", {})

    @property
    def dropped_frames(self) -> int:
        return getattr(self.writer, "dropped_frames", 0)

    def process(self):
        """Run the poll loop and close the outputs however the loop exits."""
        try:
            super().process()
        finally:
            self.close_files()

    def initialize(
        self,
        plan: AcquisitionPlan,
        core: MicroscopeCoreInterface,
        affine_transform=None,
        slm_shape=None,
    ):
        """Create the outputs for the plan. Returns the (path, layer) pairs for the GUI."""
        self.plan = plan
        return self.writer.open(plan, core, self.base_path, affine_transform, slm_shape)

    def close_files(self):
        """Finalise and close all outputs (idempotent)."""
        self.writer.close()

    def write_data(self, data: AcquisitionData):
        if isinstance(data, SegmentationData):
            self.writer.write_labels(data)
        else:
            self.writer.write_frame(data)

    def handle_data(self, data):
        aq_event = data.event

        self.write_data(data)

        if isinstance(data, SegmentationData):
            return

        if aq_event.segment:
            self.seg_queue.put(data)

        if aq_event.raw_goes_to_pattern:
            self.pattern_queue.put(data)

    def handle_message(self, msg):
        logger.info(msg)

        match msg.message:
            case "close":
                self.manager_done = True

            case "stream_close":
                self.stream_count += 1

                # First stream close (Microscope)
                if self.stream_count == 1:
                    logger.info(
                        "Outbox received stream_close from Microscope. Propagating to seg/pattern."
                    )
                    self.seg_queue.put(StreamCloseMessage())
                    self.pattern_queue.put(StreamCloseMessage())

                elif self.stream_count == 2:
                    logger.info("Outbox received stream_close from Segmentation.")

            case _:
                raise ValueError(f"Unexpected message: {msg}")

        if self.manager_done and self.stream_count >= 2:
            self.close_files()
            return True

        return False


class SLMBuffer(DataPassingProcess):
    def __init__(self, aq: AllQueues, stop_event: Event | None = None):
        super().__init__(aq, stop_event)
        self.name = "slm buffer"

        self.from_manager = aq.manager_to_slm_buffer
        self.data_in = [aq.pattern_to_slm]

        self.to_microscope = aq.slm_to_microscope

        self.slm_patterns = {}

        self.slm_shape = None
        self.affine_transform = None

        self.initialized = False

        self.manager_done = False
        self.pattern_done = False

        self.initialize_queues()

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

    def handle_data(self, data: CameraPattern):
        logger.info("SLM buffer received data from slm")

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
        Handle messages sent to the SLMBuffer from the manager
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
            "outbox": aq.manager_to_outbox,
            "slm_buffer": aq.manager_to_slm_buffer,
            "seg": aq.manager_to_seg,
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
                experiment = self.experiments[name]
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
                    segmentation_method=experiment.segmentation.method_name,
                    pattern_method=experiment.pattern.method_name,
                    binning=cfg.binning,
                    do_segmentation=ev.segment,
                    save_segmentation=ev.save_seg,
                    raw_goes_to_pattern=ev.raw_to_pattern,
                    segmentation_goes_to_pattern=ev.seg_to_pattern,
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
