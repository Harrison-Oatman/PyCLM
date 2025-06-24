from pymmcore_plus import CMMCorePlus
from .queues import AllQueues
from time import time, sleep
import numpy as np
from .events import AcquisitionEvent, UpdatePatternEvent, UpdateStagePositionEvent
from .experiments import PositionWithAutoFocus, DeviceProperty, ConfigGroup
from .datatypes import EventSLMPattern, AcquisitionData, StimulationData
from .messages import UpdateZPositionMessage
import logging

logger = logging.getLogger(__name__)


class MicroscopeProcess:

    def __init__(self, core: CMMCorePlus, aq: AllQueues):
        self.core = core
        self.inbox = aq.manager_to_microscope  # receives messages/events from manager
        self.manager = aq.microscope_to_manager  # send messages to manager
        self.outbox = aq.acquisition_outbox  # send acquisition data to outbox process
        self.slm_queue = aq.slm_to_microscope  # receives SLM updates

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
            logger.warning("SLM Device not initialized,"
                           " using dummy slm")

            self.slm_device = "dummy"
            self.slm_h = 1140
            self.slm_w = 900

        else:
            self.slm_device = dev
            self.slm_h = core.getSLMHeight(dev)
            self.slm_w = core.getSLMWidth(dev)

        self.current_pattern = np.zeros((self.slm_h, self.slm_w))

        self.slm_initialized = True

    def process(self, event_await_s=0, slm_await_s=5):

        logger.debug(f"started MicroscopeProcess on {self.core}")
        self.start = time()

        event_await_start = time()

        while True:

            if self.inbox.empty():

                # check for timeout
                if (event_await_s != 0) & (time() - event_await_start > event_await_s):
                    raise TimeoutError(f"No events in queue for {time() - event_await_start: .3f}s")

                continue

            msg = self.inbox.get()

            match msg.message:
                case "update_pattern_event":
                    self.handle_update_pattern_event(msg.event, slm_await_s)

                case "acquisition_event":
                    self.handle_acquisition_event(msg.event)

                case "update_position_event":
                    self.handle_update_position_event(msg.event)

                case "close":
                    return 0

                case _:
                    raise NotImplementedError(f"Unknown message type: {msg.message}")

            event_await_start = time()

    def handle_config_update(self, config_groups: list[ConfigGroup]):
        if config_groups is None:
            return 0

        logger.info(f"setting config groups:")

        for group, config in config_groups:
            self.core.setConfig(group, config)

            logger.info(f"{group} = {config}")

        return 0

    def handle_device_update(self, devices: list[DeviceProperty]):

        if devices is None:
            return 0

        logger.info(f"setting device properties:")

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
        except:
            return None

        binning_str = f"{binning}x{binning}"

        if binning_str in allowed:
            core.setProperty(camera, "Binning", binning_str)

        else:
            if self.warned_binning:
                return None

            logger.warning(f"attempted set binning {binning_str}, allowed binnings {allowed}")
            self.warned_binning = True

    def move_to_position(self, position: PositionWithAutoFocus) -> tuple[bool, float]:

        start_time = time()

        core = self.core

        xy = position.get_xy()
        z = position.get_z()
        pfs = position.get_autofocus_offset()

        z_moved = False

        if (z is not None) and (z < self.core.getZPosition()):
            core.setPosition(z)
            z_moved = True

        if xy is not None:
            logger.info(f"moving to xy {xy}")
            core.setXYPosition(xy[0], -xy[1])
        else:
            logger.debug(f"move_to_position called with no xy position: {position}")

        if z is not None:
            if not z_moved:
                logger.info(f"moving to z position {z}")
                core.setPosition(z)

                z_moved = True
        else:
            logger.debug(f"move_to_position called with no z position: {position}")

        if pfs is not None:
            logger.info(f"setting pfs offset {pfs}")
            core.setAutoFocusOffset(pfs)

            z_moved = True
        else:
            logger.debug(f"move_to_position called with no pfs offset: {position}")

        if not z_moved:
            return False, 0

        # todo: fix hard-coded PFS status property and PFS required on

        core.setProperty("PFS", "FocusMaintenance", "On")

        while core.getProperty("PFS", "PFS Status") != "0000001100001010":
            pass

        logger.info(f"move+focus took {time() - start_time:0.3f}")

        return True, core.getZPosition()

    def handle_update_position_event(self, up_event: UpdateStagePositionEvent):

        if isinstance(up_event.position, PositionWithAutoFocus):

            z_moved, z_new_position = self.move_to_position(up_event.position)

            if z_moved:
                old_z = up_event.position.get_z()

                if np.abs(old_z - z_new_position) > 5:
                    logger.warning( f"Major Z position change: {old_z}, {z_new_position}")

                if abs(z_new_position - old_z) > 1.0:
                    self.manager.put(
                        UpdateZPositionMessage(
                            z_new_position, up_event.experiment_name
                        )
                    )

        else:
            raise NotImplementedError("only implemented PositionWithAutoFocus")

    def handle_update_pattern_event(self, up_event: UpdatePatternEvent, slm_await_s):
        event_id = up_event.id
        logger.debug(f"handling update pattern event {event_id}")

        assert self.slm_initialized, "slm not declared to microscope process, run declare_slm first"

        pattern_data = self.slm_queue.get(True, slm_await_s)

        assert isinstance(pattern_data, EventSLMPattern), f"received pattern data of unknown type: {type(pattern_data)}"
        assert pattern_data.event_id == event_id, f"event mismatch"

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
            logger.info(f"{self.t(): .3f}| waiting {t_delta: .3f}s until next acquisition")
            sleep(t_delta)

        logger.debug("wait for system")
        wait_time = time()
        self.core.waitForSystem()
        logger.debug(f"took {time() - wait_time: .3f}s")

        sleep(1.0)

        # print(aq_event.position.get_z(), self.core.getPosition())

        logger.info(f"{self.t(): .3f}| acquiring image: {aq_event.exposure_time_ms}ms")
        image = self.snap()
        aq_event.completed_time = time()
        logger.info(f"{self.t(): .3f}| image acquired")

        aq_event.pixel_width_um = self.core.getPixelSizeUm()

        # info(f"{self.t(): .3f}| unloading")

        if aq_event.needs_slm:
            data_out = StimulationData(aq_event, image, self.current_pattern, self.current_pattern_id)
        else:
            data_out = AcquisitionData(aq_event, image)

        self.outbox.put(data_out)
        # info(f"{self.t(): .3f}| unloaded")

    def snap(self):
        core = self.core

        core.snapImage()
        image = core.getImage()

        return image

    def t(self):
        return time() - self.start

        # tagged_image = core.getTaggedImage()
        # pixels = np.reshape(tagged_image.pix,
        #                     newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        #
        # tags = tagged_image.tags
        #
        # return pixels, tags
