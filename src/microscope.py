from pycromanager import Core
from multiprocessing import Queue

class MicroscopeProcess:

    def __init__(self, core: Core, inbox: Queue, outbox: Queue, slm_queue: Queue):
        self.core = core
        self.inbox = inbox  # receives messages/events from the controller
        self.outbox = outbox  # send acquisition data to outbox process
        self.slm_queue = slm_queue  # receives SLM updates

    
