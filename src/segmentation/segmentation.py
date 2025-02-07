from ..queues import AllQueues


# todo SegmentationModel


class SegmentationProcess:

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_seg
        self.manager = aq.seg_to_manager
        self.raw_in = aq.segmentation_queue
        self.seg_out = aq.segmented_pattern
