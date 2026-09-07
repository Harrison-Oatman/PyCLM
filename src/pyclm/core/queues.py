"""
Queues connecting the pipeline processes.

All processes run as threads in one interpreter (see ``Controller.run``), so
the queues are plain ``queue.Queue`` objects: items are passed by reference,
never copied, and ``empty()`` / ``get_nowait()`` are reliable.
"""

from queue import Empty, Queue


class AllQueues:
    def __init__(self):
        # messages from manager
        self.manager_to_microscope = Queue()
        self.manager_to_outbox = Queue()
        self.manager_to_slm_buffer = Queue()
        self.manager_to_seg = Queue()
        self.manager_to_pattern = Queue()

        # messages to manager
        self.microscope_to_manager = Queue()

        # output of microscope acquisition
        self.acquisition_outbox = Queue()

        # raw data to be segmented queue
        self.outbox_to_seg = Queue()
        self.seg_to_outbox = Queue()

        # segmented for pattern generation queue
        # and raw to pattern queue
        self.seg_to_pattern = Queue()
        self.outbox_to_pattern = Queue()

        # pattern to slm buffer queue
        self.pattern_to_slm = Queue()
        self.slm_to_microscope = Queue()

        self.all_queues = [q for q in vars(self).values() if isinstance(q, Queue)]

    def close(self):
        """Discard anything left in the queues once every process has exited."""
        for queue in self.all_queues:
            while True:
                try:
                    queue.get_nowait()
                except Empty:
                    break
