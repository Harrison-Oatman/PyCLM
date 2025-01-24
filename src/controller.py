"""
The controller is the brain of the feedback loop.

It is responsible for
- managing timing
- passing messages between processes
- scheduling microscope events
"""

from queues import AllQueues


class MicroscopeOutbox:

    # grabs data from microscope, writes data to disk

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_outbox
        self.outbox = aq.acquisition_outbox
        self.seg_queue = aq.segmentation_queue

    def process(self):
        pass


class SLMBuffer:

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_slm_buffer
        self.slm_queue = aq.slm_to_microscope
        self.pattern_queue = aq.pattern_to_slm

    def process(self):
        pass


class Manager:

    def __init__(self, aq: AllQueues):
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