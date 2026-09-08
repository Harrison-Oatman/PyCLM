"""
Control queues connecting the pipeline processes.

All processes run as threads in one interpreter (see ``Controller.run``), so
the queues are plain ``queue.Queue`` objects: items are passed by reference,
never copied, and ``empty()`` / ``get_nowait()`` are reliable.

Only addressed control messages travel here (the Manager's events, the
SLM handshake). Frame-derived data (raw frames, segmentations, tracks) is
fanned out by the :class:`~pyclm.core.router.Router`, which owns one data
inbox per consumer.
"""

from queue import Empty, Queue


class AllQueues:
    def __init__(self):
        # messages from manager
        self.manager_to_microscope = Queue()
        self.manager_to_slm_buffer = Queue()
        self.manager_to_pattern = Queue()

        # messages to manager
        self.microscope_to_manager = Queue()

        # pattern to slm buffer, slm buffer to microscope (synchronous handshake)
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
