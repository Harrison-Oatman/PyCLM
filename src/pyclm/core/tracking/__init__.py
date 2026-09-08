"""Tracking methods: stable identities for labelled regions across timepoints."""

from .centroid import CentroidTracker
from .tracking import (
    TRACK_COLUMNS,
    TRACK_LABEL_DTYPE,
    TRACK_SCHEMA,
    TrackingMethod,
    TrackRow,
    Tracks,
    rows_to_table,
)

known_tracking_methods: dict[str, type[TrackingMethod]] = {
    CentroidTracker.name: CentroidTracker,
}

__all__ = [
    "TRACK_COLUMNS",
    "TRACK_LABEL_DTYPE",
    "TRACK_SCHEMA",
    "CentroidTracker",
    "TrackRow",
    "TrackingMethod",
    "Tracks",
    "known_tracking_methods",
    "rows_to_table",
]
