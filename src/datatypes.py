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


class GenericData:

    def __init__(self, data: np.ndarray):
        self.data = data


class AcquisitionData(GenericData):

    def __init__(self, event: AcquisitionEvent, data: np.ndarray):
        super().__init__(data)

        self.event = event
        self.event_id = event.id
        self.channel_id = event.channel_id


class StimulationData(AcquisitionData):

    def __init__(self, event: AcquisitionEvent, data: np.ndarray, dmd_pattern: np.ndarray,
                 pattern_id):
        super().__init__(event, data)
        self.dmd_pattern = dmd_pattern
        self.pattern_id = pattern_id


class SegmentationData(AcquisitionData):

    pass


class CameraPattern(GenericData):

    def __init__(self, experiment_name, data: np.ndarray, slm_coords=False):
        super().__init__(data)

        self.experiment = experiment_name
        self.pattern_id = uuid4()

        self.slm_coords = slm_coords
