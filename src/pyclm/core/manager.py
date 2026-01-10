"""
The controller is the brain of the feedback loop.

It is responsible for
- managing timing
- passing messages between processes
- scheduling microscope events
"""

import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import tifffile
from cv2 import warpAffine
from h5py import File

from pyclm.core.pattern_process import RequestPattern

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
    UpdatePositionWithAutoFocusEvent,
)
from .experiments import (
    Experiment,
    ExperimentSchedule,
    ImagingConfig,
    TimeCourse,
)
from .messages import (
    AcquisitionEventMessage,
    Message,
    StreamCloseMessage,
    UpdatePatternEventMessage,
    UpdatePositionEventMessage,
    UpdateZPositionMessage,
)
from .patterns import AcquiredImageRequest
from .queues import AllQueues

logger = logging.getLogger(__name__)


from threading import Event

from .base_process import BaseProcess


class DataPassingProcess(BaseProcess, metaclass=ABCMeta):
    def __init__(self, aq: AllQueues, stop_event: Event | None = None):
        super().__init__(stop_event, name="data passing process")
        self.all_queues = aq

        self.message_history = []

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
            print(self.name, data.message)
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
        self.message_history.append(msg)

        match msg.message:
            case "close":
                return True

            case _:
                raise ValueError(f"Unexpected message: {msg}")


class MicroscopeOutbox(DataPassingProcess):
    # grabs data from microscope, writes data to disk

    def __init__(
        self,
        aq: AllQueues,
        base_path: Path | None = None,
        save_type="hdf5",
        stop_event: Event | None = None,
    ):
        super().__init__(aq, stop_event)
        self.name = "microscope outbox"

        if base_path is None:
            base_path = Path().cwd()

        self.from_manager = aq.manager_to_outbox
        self.data_in = [aq.acquisition_outbox, aq.seg_to_outbox]

        self.manager = aq.outbox_to_manager

        self.manager_done = False
        self.stream_count = 0

        self.seg_queue = aq.outbox_to_seg
        self.pattern_queue = aq.outbox_to_pattern

        self.base_path = base_path
        self.save_type = save_type

        self.initialize_queues()

    def handle_data(self, data):
        aq_event = data.event

        self.write_data(data)

        if isinstance(data, SegmentationData):
            return

        # print(aq_event)

        if aq_event.segment:
            self.seg_queue.put(data)

        if aq_event.raw_goes_to_pattern:
            self.pattern_queue.put(data)

    def handle_message(self, msg):
        self.message_history.append(msg)
        logger.info(msg)

        match msg.message:
            case "close":
                self.manager_done = True

            case "stream_close":
                self.stream_count += 1

                print("outbox received stream_close")

                # First stream close (Microscope)
                if self.stream_count == 1:
                    logger.info(
                        "Outbox received stream_close from Microscope. Propagating to seg/pattern."
                    )
                    close_msg = StreamCloseMessage()
                    self.seg_queue.put(close_msg)

                    close_msg = StreamCloseMessage()
                    self.pattern_queue.put(close_msg)

                elif self.stream_count == 2:
                    logger.info("Outbox received stream_close from Segmentation.")

            case _:
                raise ValueError(f"Unexpected message: {msg}")

        if self.manager_done and self.stream_count >= 2:
            return True

        return False

    def write_data(self, data: AcquisitionData):
        aq_event = data.event

        file_relpath, relpath = aq_event.get_rel_path()

        # acquisition is saved as "data", its segmentation is saved as "seg"
        dset_name = r"data"
        if isinstance(data, SegmentationData):
            dset_name = r"seg"

        try:
            if self.save_type == "tif":
                fullpath = self.base_path / file_relpath / relpath
                fullpath.mkdir(parents=True, exist_ok=True)

                tifffile.imwrite(fullpath / "data.tif", data.data)

            else:
                filepath = self.base_path / f"{file_relpath}.hdf5"
                # Ensure directory exists
                filepath.parent.mkdir(parents=True, exist_ok=True)

                with File(filepath, "a") as f:
                    if aq_event.save_output:
                        dset = f.create_dataset(relpath + dset_name, data=data.data)
                        aq_event.write_attrs(dset)

                    if isinstance(data, StimulationData):
                        if aq_event.save_stim:
                            dset = f.create_dataset(
                                relpath + r"dmd", data=data.dmd_pattern
                            )
                            dset.attrs["pattern_id"] = str(data.pattern_id)
                            aq_event.write_attrs(dset)

        except Exception as e:
            logger.error(f"Failed to write data: {e}", exc_info=True)


class SLMBuffer(DataPassingProcess):
    def __init__(self, aq: AllQueues, stop_event: Event | None = None):
        super().__init__(aq, stop_event)
        self.name = "slm buffer"

        self.from_manager = aq.manager_to_slm_buffer
        self.data_in = [aq.pattern_to_slm]

        self.manager = aq.slm_buffer_to_manager

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
        """
        :param shape: Tuple of (height, width) for the SLM pattern
        :param affine_transform: 2x3 array for affine transformation to apply to pattern
        :param experiment_names: List of experiment names to initialize patterns for
        """

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
            print(
                f"Warning: Experiment name '{experiment_name}' not found in SLM patterns."
            )

    def handle_message(self, msg):
        """
        Handle messages sent to the SLMBuffer from the manager
        :param msg: Message object
        :return: bool indicating whether to close the process
        """
        self.message_history.append(msg)

        match msg.message:
            case "close":
                self.manager_done = True

            case "stream_close":
                self.pattern_done = True

            case "initialize_slm_queue":
                # Initialize the SLM buffer with provided parameters
                shape = msg.shape
                affine_transform = msg.affine_transform
                experiment_names = msg.experiment_names

                self.initialize(shape, affine_transform, experiment_names)

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
            "outbox": aq.outbox_to_manager,
            "slm_buffer": aq.slm_buffer_to_manager,
            "seg": aq.seg_to_manager,
            "pattern": aq.pattern_to_manager,
        }

        self.initialized = False
        self.schedule = None
        self.experiments = None
        self.times = None
        self.positions = None

        self.pattern_lcms = None
        self.pattern_requirements = None

    def initialize(
        self,
        schedule: ExperimentSchedule,
        requirements: dict[str, list[AcquiredImageRequest]],
    ):
        self.schedule = schedule
        self.experiments: dict[str, Experiment] = schedule.experiments
        self.positions = schedule.positions
        self.times = schedule.times

        self.pattern_requirements = requirements

        self.pattern_lcms = {}
        for name in self.experiments:
            self.pattern_lcms[name] = self.get_pattern_lcm(
                self.experiments[name], requirements[name]
            )

        self.initialized = True

    def get_kwargs(
        self, experiment: Experiment, channel: ImagingConfig, make_pattern: bool
    ):
        kwargs = {
            "save_output": channel.save,
            "segmentation_method": experiment.segmentation.method_name,
            "pattern_method": experiment.pattern.method_name,
            "binning": channel.binning,
            "do_segmentation": False,
            "save_segmentation": False,
            "raw_goes_to_pattern": False,
            "segmentation_goes_to_pattern": False,
        }

        print(f"channel {channel}")

        requirements = self.pattern_requirements[experiment.experiment_name]
        channel_id = channel.channel_id

        if not make_pattern:
            return kwargs

        value = None

        for air in requirements:
            air: AcquiredImageRequest

            print(f"air: {air}")

            if channel_id == air.id:
                value = air

        if value is None:
            return kwargs

        value: AcquiredImageRequest

        kwargs.update(
            {
                "do_segmentation": value.needs_seg,
                "save_segmentation": experiment.segmentation.save,
                "raw_goes_to_pattern": value.needs_raw,
                "segmentation_goes_to_pattern": value.needs_seg,
            }
        )

        return kwargs

    def handle_message(self, msg: Message):
        match msg.message:
            case "update_z_position":
                assert isinstance(msg, UpdateZPositionMessage)
                name = msg.experiment_name
                val = msg.new_z_position

                self.positions[name].z = val

            case _:
                raise ValueError(f"Unexpected message: {msg}")

    @staticmethod
    def get_pattern_lcm(
        experiment: Experiment, requirements: list[AcquiredImageRequest]
    ):
        pattern_required_channels = [r.id for r in requirements]

        t_vals = [
            c.every_t
            for c in experiment.channels.values()
            if c.channel_id in pattern_required_channels
        ]

        stim = experiment.stimulation
        if stim.channel_id in pattern_required_channels:
            t_vals.append(stim.every_t)

        t_vals.append(experiment.pattern.every_t)

        return np.lcm.reduce(np.array(t_vals, dtype=int))

    def construct_position_event_message(self, position, name):
        self.msgout["microscope"].put(
            UpdatePositionEventMessage(UpdatePositionWithAutoFocusEvent(position, name))
        )

    def send_make_pattern_request(self, loop_iter, experiment_name, time_sec):
        """

        :param loop_iter: current loop
        :param experiment_name: name of experiment
        :param time_sec: scheduled time of pattern (used by pattern generation module)
        :return:
        """
        print(experiment_name, self.pattern_lcms[experiment_name])
        make_pattern = (loop_iter % self.pattern_lcms[experiment_name]) == 0

        print(f"pattern_requirements: {self.pattern_requirements[experiment_name]}")

        if make_pattern:
            pattern_request = RequestPattern(
                loop_iter,
                time_sec,
                experiment_name,
                self.pattern_requirements[experiment_name],
            )

            self.msgout["pattern"].put(pattern_request)

        return make_pattern

    def process(self):
        assert self.initialized, (
            "manager must be initialized with an experiment schedule to start"
        )

        times: TimeCourse = self.times
        start_time = time() + times.setup

        # time iter loop
        for t in range(times.count):
            print(f"t = {t}: {(time() - start_time) / 60: 0.1f} minutes")

            # wait until preparatory phase
            # todo: check if we are behind schedule
            while (time() - start_time) < (t * times.interval) - times.setup:
                for _source, inbox in self.msgin.items():
                    while not inbox.empty():
                        msg = inbox.get()
                        self.handle_message(msg)

                if self.stop_event and self.stop_event.is_set():
                    print("force stopping manager process")
                    return

            # iterate through each experiment
            for i, (name, experiment) in enumerate(self.experiments.items()):
                scheduled_time = start_time + (t * times.interval) + (i * times.between)

                # account for scheduled delay if applicable
                t_delay = experiment.t_delay  # 0 unless specified
                if t < t_delay:
                    continue

                this_t = t - t_delay

                # check if stop early
                if experiment.t_stop > 0:
                    if this_t >= experiment.t_stop:
                        continue

                # determine and send if new pattern should be generated
                make_pattern = self.send_make_pattern_request(
                    this_t, name, scheduled_time - start_time
                )

                # tracks when to send update position message
                position_passed = False

                """Stimulation Event"""
                stim = experiment.stimulation
                if this_t % stim.every_t == 0:
                    # create update position event if first imaging condition in loop
                    if not position_passed:
                        self.construct_position_event_message(
                            self.positions[name], name
                        )
                        position_passed = True

                    if stim.exposure > 0:
                        # create update pattern and acquisition event
                        channel_kwargs = self.get_kwargs(experiment, stim, make_pattern)
                        update_pattern = UpdatePatternEvent(
                            name, stim.get_config_groups(), stim.get_device_properties()
                        )
                        pattern_acquisition = AcquisitionEvent(
                            name,
                            self.positions[name],
                            stim.channel_id,
                            scheduled_time=scheduled_time,
                            scheduled_time_since_start=scheduled_time - start_time,
                            exposure_time_ms=stim.exposure,
                            needs_slm=True,
                            config_groups=stim.get_config_groups(),
                            devices=stim.get_device_properties(),
                            sub_axes=[f"{t: 05d}", "stim_aq"],
                            t_index=t,
                            **channel_kwargs,
                        )

                        upmsg = UpdatePatternEventMessage(update_pattern)
                        aqmsg = AcquisitionEventMessage(pattern_acquisition)

                        self.msgout["slm_buffer"].put(upmsg)
                        self.msgout["microscope"].put(upmsg)
                        self.msgout["microscope"].put(aqmsg)

                """Image Acquisition Events (each channel)"""
                for channel_name, channel in experiment.channels.items():
                    if this_t % channel.every_t == 0:
                        # create update position event if first imaging condition in loop
                        if not position_passed:
                            self.construct_position_event_message(
                                self.positions[name], name
                            )
                            position_passed = True

                        # create acquisition event
                        channel_kwargs = self.get_kwargs(
                            experiment, channel, make_pattern
                        )
                        channel_acquisition = AcquisitionEvent(
                            name,
                            self.positions[name],
                            channel.channel_id,
                            scheduled_time=scheduled_time,
                            scheduled_time_since_start=scheduled_time - start_time,
                            exposure_time_ms=channel.exposure,
                            config_groups=channel.get_config_groups(),
                            devices=channel.get_device_properties(),
                            sub_axes=[f"{t: 05d}", f"channel_{channel_name}"],
                            t_index=t,
                            **channel_kwargs,
                        )

                        aqmsg = AcquisitionEventMessage(channel_acquisition)
                        self.msgout["microscope"].put(aqmsg)

        print("DONE")

        for box in self.msgout:
            msg = Message()
            msg.message = "close"
            self.msgout[box].put(msg)
