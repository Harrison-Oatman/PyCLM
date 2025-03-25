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

    def process(self):
        pass
