from pymmcore_plus import CMMCorePlus
from .queues import AllQueues
from time import time, sleep
import numpy as np
from .events import AcquisitionEvent, UpdatePatternEvent, Position
from .datatypes import EventSLMPattern
import logging


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

    def declare_slm(self):
        core = self.core
        dev = core.getSLMDevice()

        self.slm_device = dev
        self.slm_h = core.getSLMHeight(dev)
        self.slm_w = core.getSLMWidth(dev)

        self.slm_initialized = True

    def process(self, event_await_s=5, slm_await_s=5):

        logging.debug(f"started MicroscopeProcess on {self.core}")

        event_await_start = time()

        while True:

            if self.inbox.empty():

                logging.debug(f"{time() - event_await_start: .3f}s since last event")

                # check for timeout
                if (event_await_s != 0) & (time() - event_await_start > event_await_s):
                    raise TimeoutError(f"No events in queue for {time() - event_await_start: .3f}s")

                continue

            msg = self.inbox.get()

            match msg.message:
                case "update_pattern_event":
                    self.handle_update_pattern_event(msg, slm_await_s)

                case "acquisition_event":
                    self.handle_acquisition_event(msg)

                case "close":
                    return 0

                case _:
                    raise NotImplementedError(f"Unknown message type: {msg.message}")

            event_await_start = time()

    def handle_config_update(self, config_groups):
        if config_groups is None:
            return 0

        for group, config in config_groups:
            self.core.setConfig(group, config)

        return 0

    def handle_device_update(self, devices):

        if devices is None:
            return 0

        for label, name, value, t in devices:

            t_func = {
                "str": str,
                "float": float,
                "int": int,
                "bool": bool,
            }[t]

            self.core.setProperty(label, name, t_func(value))

    def move_to_position(self, position: Position):

        core = self.core

        xy = position.get_xy()
        if xy is not None:
            logging.info(f"moving to xy {xy}")
            core.setXYPosition(xy[0], xy[1])
        else:
            logging.debug(f"move_to_position called with no xy position: {position}")

        z = position.get_z()
        if z is not None:
            logging.info(f"moving to z position {z}")
            core.setPosition(z)
        else:
            logging.debug(f"move_to_position called with no z position: {position}")

        pfs = position.get_pfs()
        if pfs is not None:
            logging.info(f"setting pfs offset {pfs}")
            core.setAutoFocusOffset(pfs)
        else:
            logging.debug(f"move_to_position called with no pfs offset: {position}")

        return 0

    def handle_update_pattern_event(self, up_event: UpdatePatternEvent, slm_await_s):
        event_id = up_event.id
        logging.debug(f"handling update pattern event {event_id}")

        assert self.slm_initialized, "slm not declared to microscope process, run declare_slm first"

        self.handle_device_update(up_event.devices)
        self.handle_config_update(up_event.config_groups)

        pattern_data = self.slm_queue.get(True, slm_await_s)

        assert pattern_data is EventSLMPattern, f"received pattern data of unknown type: {type(pattern_data)}"
        assert pattern_data.event_id == event_id, f"event mismatch"

        self.core.setSLMImage(self.slm_device, pattern_data)
        logging.info(f"experiment {up_event.experiment_name}: set slm image")

        return 0

    def handle_acquisition_event(self, aq_event: AcquisitionEvent):
        event_id = aq_event.id
        logging.debug(f"handling acquisition event {event_id}")

        self.handle_device_update(aq_event.devices)
        self.handle_config_update(aq_event.config_groups)
        self.core.setExposure(aq_event.exposure_time_ms)

        self.move_to_position(aq_event.position)

        target_time = aq_event.scheduled_time
        t_delta = target_time - time() - 0.1

        if t_delta > 0:
            logging.info(f"waiting {t_delta: .3f}s until next acquisition")
            sleep(t_delta)

        logging.debug("wait for system")
        wait_time = time()
        self.core.waitForSystem()
        logging.debug(f"took {wait_time - time(): .3f}s")

        logging.info("acquiring image")
        image = self.snap()
        aq_event.completed_time = time()

    def snap(self):
        core = self.core

        core.snapImage()

        tagged_image = core.getTaggedImage()
        pixels = np.reshape(tagged_image.pix,
                            newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])

        tags = tagged_image.tags

        return pixels, tags
