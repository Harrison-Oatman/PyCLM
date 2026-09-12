import logging
from queue import Empty
from threading import Event
from time import sleep, time
from typing import ClassVar

import numpy as np

from .base_process import BaseProcess
from .core_interface import MicroscopeCoreInterface
from .datatypes import (
    AcquisitionData,
    EventSLMPattern,
    SkippedAcquisition,
    StimulationData,
)
from .events import AcquisitionEvent, UpdatePatternEvent, UpdateStagePositionEvent
from .experiments import ConfigGroup, DeviceProperty
from .grid import stitch
from .messages import EventDoneMessage, Message, UpdateZPositionMessage
from .position_mover import BasicPositionMover, PositionMover
from .queues import AllQueues

logger = logging.getLogger(__name__)


def pattern_is_blank(pattern) -> bool:
    """True when nothing in the DMD image (or any tile of a list of them) is lit."""
    if pattern is None:
        return True
    if isinstance(pattern, list | tuple):
        return all(pattern_is_blank(p) for p in pattern)
    return not np.any(np.asarray(pattern))


class MicroscopeProcess(BaseProcess):
    """
    Executes acquisition, stage and SLM events against a MicroscopeCoreInterface.

    Every frame it takes is published to the Router (it is the producer of
    ``raw``); nothing about who consumes a frame is decided here.

    An error while handling one message is logged and counted, and the process
    moves on to the next message, so a single failed event does not end the
    experiment. After ``max_consecutive_errors`` failures in a row the error is
    re-raised, which makes the Controller abort the run.
    """

    produces: ClassVar[dict[str, tuple[str, ...]]] = {"raw": (), "skipped": ()}
    always_active: ClassVar[bool] = True

    def __init__(
        self,
        core: MicroscopeCoreInterface,
        aq: AllQueues,
        position_mover: PositionMover | None = None,
        stop_event: Event | None = None,
        settle_time_s: float = 1.0,
        slm_await_s: float = 5.0,
        max_consecutive_errors: int = 10,
    ):
        super().__init__(stop_event, name="microscope")
        self.core = core
        self.position_mover = (
            position_mover if position_mover is not None else BasicPositionMover()
        )

        self.inbox = aq.manager_to_microscope  # receives messages/events from manager
        self.manager = aq.microscope_to_manager  # send messages to manager
        self.slm_queue = aq.slm_to_microscope  # receives SLM updates

        # set by Router.resolve(); frames are published through it
        self.router = None
        self._stream_ended = False

        # seconds to wait after waitForSystem() before snapping
        self.settle_time_s = settle_time_s
        # seconds to wait for the SLM buffer to answer an update_pattern_event
        self.slm_await_s = slm_await_s
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0

        # how long inbox.get() blocks before the stop event is re-checked
        self.poll_interval = 0.05

        self.slm_initialized = False
        self.slm_device = None
        self.slm_h = None
        self.slm_w = None

        self.start = 0

        self.current_pattern = None
        self.current_pattern_id = None
        self.current_camera_pattern = None
        self.current_dmd_ids = None
        # (t, experiment) whose stimulation-only visit a blank pattern cancelled
        self._blank_skips: set[tuple[int, str]] = set()
        self.skipped_moves = 0
        self.skipped_acquisitions = 0

        self.warned_binning = False

    # ------------------------------------------------------------ routing
    def subscriptions(self, plan) -> list:
        return []

    def attach(self, router, data_inbox=None):
        """Called by the Router; the microscope consumes nothing, so the inbox is unused."""
        self.router = router

    def _emit(self, data):
        if self.router is None:
            raise RuntimeError("microscope process is not attached to a router")
        self.router.publish(data)

    def end_stream(self):
        """Tell the router no more frames are coming (idempotent)."""
        if self._stream_ended or self.router is None:
            return
        self._stream_ended = True
        self.router.end_stream(self.name)

    def declare_slm(self):
        core = self.core
        dev = core.getSLMDevice()

        if dev == "":
            logger.warning("SLM Device not initialized, using dummy slm")

            self.slm_device = "dummy"
            self.slm_h = 1140
            self.slm_w = 900

        else:
            self.slm_device = dev
            self.slm_h = core.getSLMHeight(dev)
            self.slm_w = core.getSLMWidth(dev)

        self.current_pattern = np.zeros((self.slm_h, self.slm_w))

        self.slm_initialized = True

    def process(self, event_await_s=0, slm_await_s=None):
        logger.debug(f"started MicroscopeProcess on {self.core}")
        self.start = time()

        event_await_start = time()

        while True:
            if self.stop_event and self.stop_event.is_set():
                logger.info("force stopping microscope process")
                break

            try:
                msg = self.inbox.get(timeout=self.poll_interval)
            except Empty:
                if (event_await_s != 0) and (
                    time() - event_await_start > event_await_s
                ):
                    raise TimeoutError(
                        f"No events in queue for {time() - event_await_start: .3f}s"
                    ) from None
                continue

            try:
                should_stop = self.handle_message(msg, slm_await_s)
            except Exception as exc:
                self.error_count += 1
                self.consecutive_errors += 1
                logger.error(
                    f"Error handling {msg} in microscope process "
                    f"({self.consecutive_errors} consecutive, {self.error_count} total)",
                    exc_info=True,
                )
                if getattr(msg, "message", None) == "acquisition_event":
                    self.manager.put(EventDoneMessage(msg.event, error=repr(exc)))
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.critical(
                        f"{self.consecutive_errors} consecutive microscope errors; "
                        "aborting run"
                    )
                    raise
            else:
                self.consecutive_errors = 0
                if should_stop:
                    return 0

            event_await_start = time()

    def handle_message(self, msg: Message, slm_await_s: float | None = None) -> bool:
        """Dispatch one message from the manager. Returns True when the process should exit."""
        match msg.message:
            case "update_pattern_event":
                self.handle_update_pattern_event(msg.event, slm_await_s)

            case "acquisition_event":
                self.handle_acquisition_event(msg.event)

            case "update_position_event":
                self.handle_update_position_event(msg.event)

            case "close":
                # no more events from the manager: end the raw stream
                self.end_stream()
                return True

            case _:
                raise NotImplementedError(f"Unknown message type: {msg.message}")

        return False

    def handle_config_update(self, config_groups: list[ConfigGroup]):
        if config_groups is None:
            return 0

        logger.info("setting config groups:")

        for group, config in config_groups:
            self.core.setConfig(group, config)

            logger.info(f"{group} = {config}")

        return 0

    def handle_device_update(self, devices: list[DeviceProperty]):
        if devices is None:
            return 0

        logger.info("setting device properties:")

        for label, name, value, t in devices:
            t_func = {
                "str": str,
                "float": float,
                "int": int,
                "bool": bool,
            }[t]

            logger.info(f"{label}-{name}: {t} = {value}")

            self.core.setProperty(label, name, t_func(value))

        return 1

    def set_binning(self, binning: int):
        core = self.core
        camera = self.core.getCameraDevice()

        try:
            allowed = core.getAllowedPropertyValues(camera, "Binning")
        except Exception:
            return None

        binning_str = f"{binning}x{binning}"

        if binning_str in allowed:
            core.setProperty(camera, "Binning", binning_str)

        else:
            if self.warned_binning:
                return None

            logger.warning(
                f"attempted set binning {binning_str}, allowed binnings {allowed}"
            )
            self.warned_binning = True

    @staticmethod
    def _skip_key(event) -> tuple[int, str] | None:
        index = getattr(event, "index", None) or {}
        if "t" not in index:
            return None
        return (int(index["t"]), event.experiment_name)

    def _skipping(self, event) -> bool:
        """Whether this event belongs to a visit a blank pattern cancelled."""
        if not getattr(event, "skippable_if_blank", False):
            return False
        key = self._skip_key(event)
        return key is not None and key in self._blank_skips

    def handle_update_position_event(self, up_event: UpdateStagePositionEvent):
        if self._skipping(up_event):
            self.skipped_moves += 1
            logger.info(
                f"experiment {up_event.experiment_name}: blank pattern, "
                "stimulation only; stage move skipped"
            )
            return
        position = up_event.position
        # a grid is visited tile by tile inside each acquisition; the position
        # event brings the stage to its first tile
        tiles = getattr(position, "tiles", None)
        target = tiles[0] if tiles else position
        z_moved, z_new_position = self.position_mover.move_to(target, self.core)

        if z_moved:
            old_z = position.z

            if np.abs(old_z - z_new_position) > 5:
                logger.warning(f"Major Z position change: {old_z}, {z_new_position}")

            if abs(z_new_position - old_z) > 1.0:
                self.manager.put(
                    UpdateZPositionMessage(z_new_position, up_event.experiment_name)
                )

    def _await_slm_pattern(self, event_id, timeout_s: float) -> EventSLMPattern | None:
        """
        Wait for the SLM buffer's reply to ``event_id``.

        Replies for other events (left over from an earlier timeout) are
        discarded. Returns None if no matching reply arrives within ``timeout_s``.
        """
        deadline = time() + timeout_s

        while True:
            remaining = deadline - time()
            if remaining <= 0:
                return None

            try:
                pattern_data = self.slm_queue.get(True, remaining)
            except Empty:
                return None

            if not isinstance(pattern_data, EventSLMPattern):
                logger.warning(
                    f"discarding unexpected item on slm queue: {type(pattern_data)}"
                )
                continue

            if pattern_data.event_id != event_id:
                logger.warning(
                    f"discarding stale SLM pattern for event {pattern_data.event_id}"
                )
                continue

            return pattern_data

    def handle_update_pattern_event(
        self, up_event: UpdatePatternEvent, slm_await_s: float | None = None
    ):
        if slm_await_s is None:
            slm_await_s = self.slm_await_s

        event_id = up_event.id
        logger.debug(f"handling update pattern event {event_id}")

        if not self.slm_initialized:
            raise RuntimeError(
                "slm not declared to microscope process, run declare_slm first"
            )

        pattern_data = self._await_slm_pattern(event_id, slm_await_s)

        if pattern_data is None:
            logger.warning(
                f"experiment {up_event.experiment_name}: SLM buffer did not answer "
                f"within {slm_await_s}s; keeping the current pattern "
                f"(id {self.current_pattern_id})"
            )
            return 0

        pattern = pattern_data.pattern

        if getattr(up_event, "skippable_if_blank", False):
            key = self._skip_key(up_event)
            if key is not None:
                if pattern_is_blank(pattern):
                    self._blank_skips.add(key)
                else:
                    self._blank_skips.discard(key)
                # keep the set small: earlier timepoints are over
                self._blank_skips = {k for k in self._blank_skips if k[0] >= key[0] - 1}

        # a grid experiment's pattern is one DMD image per tile; the first goes
        # up now, the others as the tiles are visited
        first = pattern[0] if isinstance(pattern, list | tuple) else pattern
        if self.slm_device == "dummy":
            logger.info(f"experiment {up_event.experiment_name}: dummy slm set image")
        else:
            self.core.setSLMImage(self.slm_device, first)
            logger.info(f"experiment {up_event.experiment_name}: set slm image")

        self.current_pattern = pattern
        self.current_pattern_id = pattern_data.pattern_unique_id
        self.current_camera_pattern = pattern_data.camera_pattern
        self.current_dmd_ids = pattern_data.dmd_ids

        return 0

    def handle_acquisition_event(self, aq_event: AcquisitionEvent):
        if self._skipping(aq_event):
            self.skipped_acquisitions += 1
            aq_event.completed_time = time()
            logger.info(
                f"experiment {aq_event.experiment_name} t={aq_event.t_index}: "
                "stimulation skipped (blank pattern)"
            )
            # the writer learns of it on the data stream, in order with the frames
            self._emit(SkippedAcquisition(aq_event))
            self.manager.put(EventDoneMessage(aq_event, skipped=True))
            return
        event_id = aq_event.id
        logger.debug(f"{self.t(): .3f}| handling acquisition event {event_id}")

        self.handle_config_update(aq_event.config_groups)
        self.handle_device_update(aq_event.devices)
        self.core.setExposure(aq_event.exposure_time_ms)

        self.set_binning(aq_event.binning)

        target_time = aq_event.scheduled_time
        t_delta = target_time - time() - 0.1

        if t_delta > 0:
            logger.info(
                f"{self.t(): .3f}| waiting {t_delta: .3f}s until next acquisition"
            )
            sleep(t_delta)

        tiles = getattr(aq_event.position, "tiles", None)
        geometry = getattr(aq_event.position, "geometry", None)
        if tiles and geometry is not None:
            image = self._acquire_grid(aq_event, tiles, geometry)
        else:
            self._settle()
            logger.info(
                f"{self.t(): .3f}| acquiring image: {aq_event.exposure_time_ms}ms"
            )
            image = self.snap()
        aq_event.completed_time = time()
        logger.info(f"{self.t(): .3f}| image acquired")

        aq_event.pixel_width_um = self.core.getPixelSizeUm()

        if aq_event.needs_slm:
            data_out = StimulationData(
                aq_event,
                image,
                self.current_pattern,
                self.current_pattern_id,
                camera_pattern=self.current_camera_pattern,
                dmd_ids=self.current_dmd_ids,
            )
        else:
            data_out = AcquisitionData(aq_event, image)

        self._emit(data_out)
        # acknowledge to the Manager: lateness and errors are tracked there
        self.manager.put(EventDoneMessage(aq_event))

    def _settle(self):
        logger.debug("wait for system")
        wait_time = time()
        self.core.waitForSystem()
        logger.debug(f"took {time() - wait_time: .3f}s")
        if self.settle_time_s > 0:
            sleep(self.settle_time_s)

    def _acquire_grid(self, aq_event: AcquisitionEvent, tiles, geometry):
        """
        Visit every tile of a grid position for one channel: move, settle,
        snap (with the tile's own DMD image up for a stimulation frame), then
        stitch the tiles into the one frame the pipeline consumes.
        """
        patterns = self.current_pattern if aq_event.needs_slm else None
        per_tile = isinstance(patterns, list | tuple)
        frames = []
        for k, tile in enumerate(tiles):
            if per_tile and self.slm_device != "dummy":
                self.core.setSLMImage(self.slm_device, patterns[k])
            self.position_mover.move_to(tile, self.core)
            self._settle()
            logger.info(
                f"{self.t(): .3f}| tile {k + 1}/{len(tiles)} of "
                f"{aq_event.experiment_name}: {aq_event.exposure_time_ms}ms"
            )
            frames.append(self.snap())
        return stitch(frames, geometry, aq_event.binning)

    def snap(self):
        core = self.core

        core.snapImage()
        image = core.getImage()

        return image

    def t(self):
        return time() - self.start
