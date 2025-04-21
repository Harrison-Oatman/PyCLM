from ..queues import AllQueues


class PatternModel:

    def __init__(self, experiment_name):
        self.experiment = experiment_name

    def generate(self):
        pass


class PatternProcess:

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_pattern
        self.manager = aq.pattern_to_manager
        self.slm = aq.pattern_to_slm

    def process(self):
        pass


# todo: what is the dynamic range of patterns? [0-1] float32?
# todo: dummy model
# todo: register model
# todo: intialize pattern process with pattern shape
# todo: handle raw and segmentation input
