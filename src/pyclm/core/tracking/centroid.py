"""
The built-in tracker: nearest-centroid linking with gap closing, gated by a
maximum displacement in micrometers. An object that segmentation misses for
up to ``max_gap_frames`` frames keeps its id when it comes back; one that is
not matched starts a new track. No divisions.

Each frame is one assignment problem between the tracks still remembered
(seen last frame, or missing for at most ``max_gap_frames``) and the new
objects, with a "no link" option at the gate cost for every track and every
object (the augmented matrix of Jaqaman et al. 2008, frame to frame). Only
pairs within the gate enter it, as a sparse matrix, so the cost grows with
the number of nearby pairs rather than with the square of the field.
docs/tracking-gap-closing-design.md has the measurements.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

from .tracking import TRACK_LABEL_DTYPE, TrackingMethod, TrackRow

# stands in for a zero cost: a sparse matrix drops explicit zeros
_EPS = 1e-9


def link(previous: np.ndarray, current: np.ndarray, gate: float):
    """
    Pair rows of ``previous`` (n, 2) with rows of ``current`` (m, 2) so the
    summed distance is smallest, where a pair may be at most ``gate`` apart
    and leaving a point unpaired costs ``gate``. Returns the paired indices
    ``(i_previous, i_current)``.
    """
    n, m = len(previous), len(current)
    empty = np.zeros(0, dtype=np.int64)
    if n == 0 or m == 0:
        return empty, empty
    near = cKDTree(current).query_ball_point(previous, gate)
    ri = np.repeat(np.arange(n), [len(p) for p in near])
    if len(ri) == 0:
        return empty, empty
    ci = np.concatenate([np.asarray(p, dtype=np.int64) for p in near])
    d = np.linalg.norm(previous[ri] - current[ci], axis=1)

    # rows: tracks, then one "starts here" row per object;
    # columns: objects, then one "ends here" column per track
    rows = np.concatenate([ri, np.arange(n), n + np.arange(m), n + ci])
    cols = np.concatenate([ci, m + np.arange(n), np.arange(m), m + ri])
    cost = np.concatenate(
        [
            d + _EPS,
            np.full(n, gate + _EPS),
            np.full(m, gate + _EPS),
            np.full(len(ri), _EPS),
        ]
    )
    matrix = coo_matrix((cost, (rows, cols)), shape=(n + m, n + m)).tocsr()
    row_ind, col_ind = min_weight_full_bipartite_matching(matrix)
    order = np.argsort(row_ind)[:n]
    tracks, objects = row_ind[order], col_ind[order]
    paired = objects < m
    return tracks[paired], objects[paired]


class CentroidTracker(TrackingMethod):
    """
    Links each region to the nearest remembered track when the centroid
    moved less than ``max_distance_um``; every region gets at most one
    partner. A track whose object is missing keeps its last position and
    its id for ``max_gap_frames`` frames of the channel (0: forget it at
    once).
    """

    name = "centroid"

    def __init__(
        self,
        experiment_name: str,
        channel: str,
        max_distance_um: float = 20.0,
        max_gap_frames: int = 2,
        **kwargs,
    ):
        super().__init__(experiment_name, channel, **kwargs)
        self.max_distance_um = float(max_distance_um)
        self.max_gap_frames = int(max_gap_frames)
        if self.max_gap_frames < 0:
            raise ValueError(f"max_gap_frames must be 0 or more, not {max_gap_frames}")
        # the remembered tracks: id, last centroid (pixels), last frame seen
        self._ids = np.zeros(0, dtype=np.int64)
        self._centroids = np.zeros((0, 2), dtype=float)
        self._last_seen = np.zeros(0, dtype=np.int64)
        self._frame = -1
        self._next_id = 1

    def shift(self, dy_px, dx_px):
        self._centroids = self._centroids + np.array([float(dy_px), float(dx_px)])

    def track(self, labels, t, pixel_size_um):
        self._frame += 1
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
        gate_px = self.max_distance_um / float(pixel_size_um)
        tracks, objects = link(self._centroids, centroids, gate_px)
        ids[objects] = self._ids[tracks]
        new = ids == 0
        ids[new] = np.arange(self._next_id, self._next_id + int(new.sum()))
        self._next_id += int(new.sum())

        # remember this frame's objects and the tracks still within the gap
        waiting = np.ones(len(self._ids), dtype=bool)
        waiting[tracks] = False
        waiting &= self._frame - self._last_seen <= self.max_gap_frames
        self._ids = np.concatenate([self._ids[waiting], ids])
        self._centroids = np.concatenate([self._centroids[waiting], centroids])
        self._last_seen = np.concatenate(
            [self._last_seen[waiting], np.full(len(ids), self._frame)]
        )

        lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int64)
        lut[found] = ids
        relabelled = lut[labels].astype(TRACK_LABEL_DTYPE)

        out = [
            TrackRow(int(tid), int(lab), float(c[0]), float(c[1]), int(a), 0)
            for tid, lab, c, a in zip(ids, found, centroids, areas, strict=True)
        ]
        return relabelled, out
