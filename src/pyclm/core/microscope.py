import logging
from queue import Empty
from threading import Event
from time import sleep, time

import numpy as np

from .base_process import BaseProcess
from .core_interface import MicroscopeCoreInterface
from .datatypes import AcquisitionData, EventSLMPattern, StimulationData
from .events import AcquisitionEvent, UpdatePatternEvent, UpdateStagePositionEvent
from .experiments import ConfigGroup, DeviceProperty
from .messages import Message, StreamCloseMessage, UpdateZPositionMessage
from .position_mover import BasicPositionMover, PositionMover
from .queues import AllQueues

logger = logging.getLogger(__name__)


class MicroscopeProcess(BaseProcess):
    """
    Executes acquisition, stage and SLM events against a MicroscopeCoreInterface.

    An error while handling one message is logged and counted, and the process
    moves on to the next message, so a single failed event does not end the
    experiment. After ``max_consecutive_errors`` failures in a row the error is
    re-raised, which makes the Controller abort the run.
    """

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
        self.outbox = aq.acquisition_outbox  # send acquisition data to outbox process
        self.slm_queue = aq.slm_to_microscope  # receives SLM updates

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

        self.warned_binning = False

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
            except Exception:
                self.error_count += 1
                self.consecutive_errors += 1
                logger.error(
                    f"Error handling {msg} in microscope process "
                    f"({self.consecutive_errors} consecutive, {self.error_count} total)",
                    exc_info=True,
                )
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
                # Send stream close to outbox
                self.outbox.put(StreamCloseMessage())
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

    def handle_update_position_event(self, up_event: UpdateStagePositionEvent):
        position = up_event.position
        z_moved, z_new_position = self.position_mover.move_to(position, self.core)

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

        if self.slm_device == "dummy":
            logger.info(f"experiment {up_event.experiment_name}: dummy slm set image")
        else:
            self.core.setSLMImage(self.slm_device, pattern)
            logger.info(f"experiment {up_event.experiment_name}: set slm image")

        self.current_pattern = pattern
        self.current_pattern_id = pattern_data.pattern_unique_id

        return 0

    def handle_acquisition_event(self, aq_event: AcquisitionEvent):
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

        logger.debug("wait for system")
        wait_time = time()
        self.core.waitForSystem()
        logger.debug(f"took {time() - wait_time: .3f}s")

        if self.settle_time_s > 0:
            sleep(self.settle_time_s)

        logger.info(f"{self.t(): .3f}| acquiring image: {aq_event.exposure_time_ms}ms")
        image = self.snap()
        aq_event.completed_time = time()
        logger.info(f"{self.t(): .3f}| image acquired")

        aq_event.pixel_width_um = self.core.getPixelSizeUm()

        if aq_event.needs_slm:
            data_out = StimulationData(
                aq_event, image, self.current_pattern, self.current_pattern_id
            )
        else:
            data_out = AcquisitionData(aq_event, image)

        self.outbox.put(data_out)

    def snap(self):
        core = self.core

        core.snapImage()
        image = core.getImage()

        return image

    def t(self):
        return time() - self.start
