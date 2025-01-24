from pycromanager import Core
from queues import AllQueues
from time import time


class MicroscopeProcess:

    def __init__(self, core: Core, aq: AllQueues):
        self.core = core
        self.inbox = aq.manager_to_microscope  # receives messages/events from the controller
        self.outbox = aq.acquisition_outbox  # send acquisition data to outbox process
        self.slm_queue = aq.slm_to_microscope  # receives SLM updates

    def process(self, event_await_s=5, slm_await_s=5):

        event_await_start = time()

        while True:

            if self.inbox.empty():

                # check for timeout
                if (event_await_s != 0) & (time() - event_await_start > event_await_s):
                    raise TimeoutError(f"No events in queue for {time() - event_await_start: .3f}s")

                continue

            msg = self.inbox.get()

            if msg.message == "acquisition_event":
                self.handle_acquisition_event(msg, slm_await_s)

            elif msg.message == "close":
                return 0

            else:
                raise NameError(f"Unknown message type: {msg.message}")

            event_await_start = time()

    def handle_acquisition_event(self, aq_event, slm_await_s):
        pass