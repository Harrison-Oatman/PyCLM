"""
Tracking: stable identities for labelled regions across timepoints.

A :class:`TrackingMethod` receives one label image per acquired frame of a
channel (cells, organoids, tissue regions: whatever the segmentation method
labels) and returns the same image relabelled with track ids plus one
:class:`TrackRow` per object. The interface is "labelled regions in,
relabelled regions and a table out"; the method decides what a region is
and keeps whatever state it needs between calls, the way segmentation
methods do.

See docs/stage3-router-design.md §5.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pyarrow as pa

from ..measure import Regions

TRACK_COLUMNS = ("track_id", "label", "y", "x", "area", "parent")
TRACK_SCHEMA = pa.schema(
    [
        ("track_id", pa.int64()),
        ("label", pa.int64()),
        ("y", pa.float64()),
        ("x", pa.float64()),
        ("area", pa.int64()),
        ("parent", pa.int64()),
    ]
)
# dtype of relabelled images: ids keep growing for the whole run
TRACK_LABEL_DTYPE = np.uint32


class TrackRow(NamedTuple):
    """One tracked object at one timepoint. ``y``/``x`` are the centroid in pixels of the (binned) frame."""

    track_id: int
    label: int
    y: float
    x: float
    area: int
    parent: int = 0


def rows_to_table(rows) -> pa.Table:
    rows = list(rows)
    return pa.table(
        {col: [getattr(r, col) for r in rows] for col in TRACK_COLUMNS},
        schema=TRACK_SCHEMA,
    )


class Tracks(Regions):
    """
    What a pattern method sees for one channel at one timepoint: the label
    image relabelled with track ids (``labels``) and the per-object rows.
    A :class:`~pyclm.core.measure.Regions`, so ``measure``, ``paint``,
    ``select`` and ``owner_of`` work on tracked objects too.
    """

    def __init__(self, labels: np.ndarray, rows):
        super().__init__(labels)
        self.rows: list[TrackRow] = list(rows)

    @property
    def ids(self) -> np.ndarray:
        return np.array([r.track_id for r in self.rows], dtype=np.int64)

    @property
    def table(self) -> pa.Table:
        return rows_to_table(self.rows)

    def to_pandas(self):
        return self.table.to_pandas()

    def row(self, track_id: int) -> TrackRow | None:
        for r in self.rows:
            if r.track_id == track_id:
                return r
        return None

    def centroid(self, track_id: int) -> tuple[float, float] | None:
        r = self.row(track_id)
        return None if r is None else (r.y, r.x)

    def mask(self, track_id: int) -> np.ndarray:
        return self.labels == track_id

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"Tracks({len(self.rows)} objects, labels {self.labels.shape})"


class TrackingMethod:
    """
    Base class for tracking methods. Subclass, set ``name``, implement
    :meth:`track`, and register with ``Controller.register_tracking_method``
    or ``run_pyclm(tracking_methods={...})``. Keyword arguments come from the
    ``[tracking]`` table of the experiment TOML (all keys except ``method``
    and ``save``).
    """

    name = "base"

    def __init__(self, experiment_name: str, channel: str, **kwargs):
        self.experiment_name = experiment_name
        self.channel = channel

    def track(
        self, labels: np.ndarray, t: int, pixel_size_um: float
    ) -> tuple[np.ndarray, list[TrackRow]]:
        """
        Link ``labels`` (one label image, 0 = background) to the objects seen
        before. Returns the image relabelled with track ids and the rows for
        this timepoint.
        """
        raise NotImplementedError(f"{self.name} does not implement track()")
