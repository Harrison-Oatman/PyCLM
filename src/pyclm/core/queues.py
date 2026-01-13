import multiprocessing
from multiprocessing import Queue

from tqdm import tqdm


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
        self.outbox_to_manager = Queue()
        self.slm_buffer_to_manager = Queue()
        self.seg_to_manager = Queue()
        self.pattern_to_manager = Queue()

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

        self.all_queues = []

        for _k, v in vars(self).items():
            if isinstance(v, multiprocessing.queues.Queue):
                self.all_queues.append(v)

    def close(self):
        print("closing all queues:")
        for queue in self.all_queues:
            assert isinstance(queue, multiprocessing.queues.Queue)

            queue.cancel_join_thread()
            queue.close()
            queue.join_thread()
