"""
Contains classes for passing structured data between processes
"""


import numpy as np
from .events import AcquisitionEvent


class EventSLMPattern:

    def __init__(self, event_id, pattern):
        self.event_id = event_id
        self.pattern = pattern


class AcquisitionData:

    def __init__(self, event: AcquisitionEvent, data: np.ndarray):
        self.event = event
        self.event_id = event.id
        self.data = data
