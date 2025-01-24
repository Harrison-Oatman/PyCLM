from multiprocessing import Queue


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
        self.segmentation_queue = Queue()

        # segmented for pattern generation queue
        # and raw to pattern queue
        self.segmented_pattern = Queue()
        self.raw_pattern = Queue()

        # pattern to slm buffer queue
        self.pattern_to_slm = Queue()
        self.slm_to_microscope = Queue()
