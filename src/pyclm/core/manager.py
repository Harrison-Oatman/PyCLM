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
from .plan import PLAN_FORMAT, AcquisitionPlan, PlannedEvent
from .queues import AllQueues

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


def get_image_shape(core: MicroscopeCoreInterface, binning: int = 1) -> tuple[int, int]:
    roi = core.getROI()
    h, w = roi[3], roi[2]
    # return (h, w)
    return (h // binning, w // binning)


class MicroscopeOutbox(DataPassingProcess):
    # grabs data from microscope, writes data to disk

    def __init__(
        self,
        aq: AllQueues,
        base_path: Path | None = None,
        stop_event: Event | None = None,
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

        self.open_files = {}  # Map experiment name to open h5py File object
        self.experiments = {}
        self.plan: AcquisitionPlan | None = None

        # frames that could not be written because no dataset was pre-allocated
        self.dropped_frames = 0

        self.initialize_queues()

    def process(self):
        """Run the poll loop and close the HDF5 files however the loop exits."""
        try:
            super().process()
        finally:
            self.close_files()

    def initialize(self, plan: AcquisitionPlan, core: MicroscopeCoreInterface):
        """
        Initialize the output files for the plan.
        Opens files in SWMR mode and writes metadata, and pre-allocates one
        dataset per frame the plan will produce.
        """
        self.plan = plan
        schedule = plan.schedule
        metadata = schedule.as_dict()
        plan_yaml = plan.yaml_str()

        try:
            for exp_name in plan.experiments:
                filepath = self.base_path / f"{exp_name}.hdf5"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if filepath.exists():
                    raise FileExistsError(
                        f"HDF5 file with this name already exists: {filepath}"
                    )

                f = File(filepath, "w", libver="latest")

                experiment = schedule.experiments[exp_name]
                self.experiments[exp_name] = experiment

                f.attrs["schedule_metadata"] = json.dumps(metadata, default=str)
                f.attrs["experiment_metadata"] = json.dumps(
                    experiment.as_dict(), default=str
                )
                f.attrs["plan"] = plan_yaml
                f.attrs["plan_format"] = PLAN_FORMAT

                # Create t_index tracker
                f.create_dataset("current_t_index", data=np.int32(-1))

                # Store every_t per storage group, plus the experiment's window
                stim_name = plan.stim_channel(exp_name)
                every_t_map = {
                    storage_group(c, c == stim_name): plan.every_t(exp_name, c)
                    for c in plan.channels(exp_name)
                }
                f.attrs["every_t"] = json.dumps(every_t_map)
                f.attrs["t_delay"] = experiment.t_delay
                f.attrs["t_stop"] = experiment.t_stop
                f.attrs["t_count"] = plan.timepoints

                self.open_files[exp_name] = f

            slm_device = core.getSLMDevice()
            slm_shape = None
            if slm_device:
                slm_shape = (
                    core.getSLMHeight(slm_device),
                    core.getSLMWidth(slm_device),
                )

            for ds in plan.expected_datasets():
                f = self.open_files[ds.experiment]
                experiment = self.experiments[ds.experiment]
                shape = get_image_shape(core, ds.config.binning)
                prefix = f"{ds.t:05d}/{ds.group}"

                self._create_frame_dataset(
                    f, f"{prefix}/data", shape, np.uint16, ds.config
                )
                if experiment.segmentation.save:
                    self._create_frame_dataset(
                        f, f"{prefix}/seg", shape, np.uint16, ds.config
                    )
                if ds.is_stim and slm_shape is not None:
                    self._create_frame_dataset(
                        f, f"{prefix}/dmd", slm_shape, np.uint8, ds.config
                    )

            # Enable SWMR only after all datasets exist
            for exp_name, f in self.open_files.items():
                f.swmr_mode = True
                logger.info(f"Initialized HDF5 file for {exp_name} in SWMR mode.")

        except Exception as e:
            logger.error(f"Failed to initialize outbox files: {e}", exc_info=True)
            self.close_files()
            raise e

        all_layers = []
        for exp_name, experiment in schedule.experiments.items():
            filepath = str((self.base_path / f"{exp_name}.hdf5").resolve())
            for channel_name in experiment.channels:
                all_layers.append((filepath, f"channel_{channel_name}"))

            if experiment.stimulation.save:
                all_layers.append((filepath, "stim_aq"))

        return all_layers

    def _create_frame_dataset(self, f, path, maxshape, dtype, config: ImagingConfig):
        dset = f.create_dataset(
            path, shape=(0, 0), maxshape=maxshape, dtype=dtype, chunks=True
        )
        self._preallocate_attrs(dset, config)

    def close_files(self):
        """Close all open HDF5 files."""
        for name, f in self.open_files.items():
            try:
                f.close()
                logger.info(f"Closed HDF5 file for {name}")
            except Exception as e:
                logger.error(f"Error closing file for {name}: {e}")
        self.open_files.clear()

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
                    close_msg = StreamCloseMessage()
                    self.seg_queue.put(close_msg)

                    close_msg = StreamCloseMessage()
                    self.pattern_queue.put(close_msg)

                elif self.stream_count == 2:
                    logger.info("Outbox received stream_close from Segmentation.")

            case _:
                raise ValueError(f"Unexpected message: {msg}")

        if self.manager_done and self.stream_count >= 2:
            self.close_files()
            return True

        return False

    def _preallocate_attrs(self, dset, channel: ImagingConfig):
        dset.attrs["id"] = ""
        dset.attrs["position"] = [("", "")]
        dset.attrs["experiment_name"] = ""
        dset.attrs["time_scheduled"] = ""
        dset.attrs["time_since_start"] = ""
        dset.attrs["time_completed"] = ""
        dset.attrs["complete"] = False
        dset.attrs["exposure_time_ms"] = 0.0
        dset.attrs["needs_slm"] = False
        dset.attrs["binning"] = 1
        dset.attrs["index"] = ""
        dset.attrs["save_output"] = False
        dset.attrs["segment"] = False
        dset.attrs["seg_method"] = ""
        dset.attrs["save_seg"] = False
        dset.attrs["raw_goes_to_pattern"] = False
        dset.attrs["seg_goes_to_pattern"] = False
        dset.attrs["channel_id"] = ""
        dset.attrs["pattern_method"] = ""
        dset.attrs["save_pattern"] = False
        dset.attrs["pixel_width_um"] = ""
        dset.attrs["pattern_id"] = ""

        for cg in channel.get_config_groups():
            dset.attrs[f"config_groups: {cg.group}"] = ""
        for dp in channel.get_device_properties():
            dset.attrs[f"devices: {dp.device}-{dp.property}"] = ""

    def _timepoint_complete(self, f, t_index: int, exp_name: str) -> bool:
        """Whether every saved frame the plan expects at ``t_index`` has been written."""
        for ds in self.plan.datasets_at(exp_name, t_index):
            if not ds.save:
                continue
            path = f"{ds.t:05d}/{ds.group}/data"
            try:
                if f[path].shape == (0, 0):
                    return False
            except KeyError:
                return False

        return True

    def write_data(self, data: AcquisitionData):
        aq_event = data.event
        relpath = aq_event.get_rel_path()

        # acquisition is saved as "data", its segmentation is saved as "seg"
        dset_name = "data"
        if isinstance(data, SegmentationData):
            dset_name = "seg"

        try:
            # Retrieve open file handle
            exp_name = aq_event.experiment_name

            if exp_name in self.open_files:
                f = self.open_files[exp_name]

                if aq_event.save_output:
                    if (relpath + dset_name) not in f:
                        self.dropped_frames += 1
                        logger.error(
                            f"no pre-allocated dataset {relpath + dset_name} in "
                            f"{exp_name}.hdf5; frame dropped "
                            f"({self.dropped_frames} dropped so far)"
                        )
                        return

                    dset = f[relpath + dset_name]
                    if dset.shape != data.data.shape:
                        dset.resize(data.data.shape)
                    for _attempt in range(3):
                        try:
                            dset[...] = data.data
                            aq_event.write_attrs(dset)
                            f.flush()
                            break
                        except PermissionError:
                            sleep(0.05)

                if isinstance(data, StimulationData):
                    if aq_event.save_stim and (relpath + "dmd") in f:
                        dset = f[relpath + "dmd"]
                        if dset.shape != data.dmd_pattern.shape:
                            dset.resize(data.dmd_pattern.shape)
                        for _attempt in range(3):
                            try:
                                dset[...] = data.dmd_pattern
                                dset.attrs["pattern_id"] = str(data.pattern_id)
                                aq_event.write_attrs(dset)
                                f.flush()
                                break
                            except PermissionError:
                                sleep(0.05)

                # Update t_index if this timepoint is newer and all scheduled channels are written
                t_index = aq_event.t_index
                if f["current_t_index"][()] < t_index and self._timepoint_complete(
                    f, t_index, exp_name
                ):
                    for _attempt in range(3):
                        try:
                            f["current_t_index"][...] = np.int32(t_index)
                            f.flush()
                            break
                        except PermissionError:
                            sleep(0.05)

            else:
                logger.warning(f"No open file found for experiment: {exp_name}")

        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to write data: {e}", exc_info=True)


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
