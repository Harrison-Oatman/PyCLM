"""
Contains classes for passing structured data between processes.

Frame-derived data carries the acquisition event it came from and a
``kind`` (``"raw"``, ``"seg"``, ``"tracks"``) that the Router uses to look up
its subscribers.
"""

from typing import ClassVar
from uuid import uuid4

import numpy as np

from .events import AcquisitionEvent
from .kinds import DEFAULT_SEGMENTATION, seg_kind


class EventSLMPattern:
    def __init__(self, event_id, pattern, pattern_unique_id=0):
        self.event_id = event_id
        self.pattern = pattern

        self.pattern_unique_id = pattern_unique_id


class GenericData:
    def __init__(self, data: np.ndarray):
        self.data = data


class AcquisitionData(GenericData):
    """A raw camera frame and the event that produced it."""

    kind: ClassVar[str] = "raw"

    def __init__(self, event: AcquisitionEvent, data: np.ndarray):
        super().__init__(data)

        self.event = event
        self.event_id = event.id
        self.channel_id = event.channel_id


class StimulationData(AcquisitionData):
    """A camera frame taken during stimulation, with the DMD pattern that was applied."""

    def __init__(
        self,
        event: AcquisitionEvent,
        data: np.ndarray,
        dmd_pattern: np.ndarray,
        pattern_id,
    ):
        super().__init__(event, data)
        self.dmd_pattern = dmd_pattern
        self.pattern_id = pattern_id


class SegmentationData(AcquisitionData):
    """
    A label image for one frame (``data``), sharing the frame's event.
    ``name`` says which ``[segmentation]`` table produced it; the routing
    ``kind`` is ``"seg"`` for the default table and ``"seg:<name>"`` otherwise.
    """

    kind: ClassVar[str] = "seg"

    def __init__(
        self,
        event: AcquisitionEvent,
        data: np.ndarray,
        name: str = DEFAULT_SEGMENTATION,
    ):
        super().__init__(event, data)
        self.name = name or DEFAULT_SEGMENTATION
        self.kind = seg_kind(self.name)


class TrackingData(AcquisitionData):
    """
    Tracking output for one frame: ``data`` (alias ``labels``) is the label
    image relabelled with stable track ids; ``rows`` are the per-object
    records for this timepoint (see ``pyclm.core.tracking.TrackRow``).
    """

    kind: ClassVar[str] = "tracks"

    def __init__(self, event: AcquisitionEvent, labels: np.ndarray, rows: list):
        super().__init__(event, labels)
        self.rows = list(rows)

    @property
    def labels(self) -> np.ndarray:
        return self.data


class CameraPattern(GenericData):
    def __init__(self, experiment_name, data: np.ndarray, slm_coords=False, binning=1):
        super().__init__(data)

        self.experiment = experiment_name
        self.pattern_id = uuid4()

        self.slm_coords = slm_coords
        self.binning = binning
