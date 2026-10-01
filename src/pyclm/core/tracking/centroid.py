"""
The built-in tracker: nearest-centroid linking with a Hungarian assignment,
gated by a maximum displacement in micrometres. No divisions, no gap
closing; an object that is not matched starts a new track and one that
disappears simply ends.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops_table

from .tracking import TRACK_LABEL_DTYPE, TrackingMethod, TrackRow


class CentroidTracker(TrackingMethod):
    """
    Links each region to the nearest region of the previous timepoint when
    the centroid moved less than ``max_distance_um``; every region gets at
    most one partner (Hungarian assignment on centroid distance).
    """

    name = "centroid"

    def __init__(
        self,
        experiment_name: str,
        channel: str,
        max_distance_um: float = 20.0,
        **kwargs,
    ):
        super().__init__(experiment_name, channel, **kwargs)
        self.max_distance_um = float(max_distance_um)
        self._prev_ids = np.zeros(0, dtype=np.int64)
        self._prev_centroids = np.zeros((0, 2), dtype=float)
        self._next_id = 1

    def track(self, labels, t, pixel_size_um):
        labels = np.asarray(labels)
        props = regionprops_table(
            labels.astype(np.int64), properties=("label", "centroid", "area")
        )
        found = np.asarray(props["label"], dtype=np.int64)
        centroids = (
            np.stack([props["centroid-0"], props["centroid-1"]], axis=1)
            if len(found)
            else np.zeros((0, 2), dtype=float)
        )
        areas = np.asarray(props["area"], dtype=np.int64)

        ids = np.zeros(len(found), dtype=np.int64)
        if len(self._prev_ids) and len(found):
            distance = cdist(self._prev_centroids, centroids) * float(pixel_size_um)
            rows, cols = linear_sum_assignment(distance)
            for r, c in zip(rows, cols, strict=True):
                if distance[r, c] <= self.max_distance_um:
                    ids[c] = self._prev_ids[r]
        for i in range(len(ids)):
            if ids[i] == 0:
                ids[i] = self._next_id
                self._next_id += 1

        lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int64)
        lut[found] = ids
        relabelled = lut[labels].astype(TRACK_LABEL_DTYPE)

        out = [
            TrackRow(int(tid), int(lab), float(c[0]), float(c[1]), int(a), 0)
            for tid, lab, c, a in zip(ids, found, centroids, areas, strict=True)
        ]
        self._prev_ids = ids
        self._prev_centroids = centroids
        return relabelled, out
