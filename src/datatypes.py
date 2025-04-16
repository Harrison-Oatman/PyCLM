"""
Contains classes for passing structured data between processes
"""

from uuid import uuid4
import numpy as np
from .events import AcquisitionEvent


class EventSLMPattern:

    def __init__(self, event_id, pattern, pattern_unique_id=0):
        self.event_id = event_id
        self.pattern = pattern

        self.pattern_unique_id = pattern_unique_id


class AcquisitionData:

    def __init__(self, event: AcquisitionEvent, data: np.ndarray):
        self.event = event
        self.event_id = event.id
        self.data = data


class CameraPattern:

    def __init__(self, event: AcquisitionEvent, data: np.ndarray):
        self.event = event
        self.experiment = event.experiment_name
        self.data = data

        self.pattern_id = uuid4()
