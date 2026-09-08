"""
The measurement toolbox for pattern methods.

Three small things that every per-cell method otherwise re-implements:

- :class:`Regions`: a label image with per-object helpers. ``measure``
  gives one value per object, ``paint`` turns per-object values back into
  an image, ``owner_of`` matches the objects of one segmentation (nuclei)
  to those of another (cells). A :class:`~pyclm.core.tracking.Tracks` is a
  ``Regions`` whose ids persist across timepoints.
- :class:`PerTrack`: a value remembered per track id, read back as an array
  aligned with the ids you ask for, with a population default for ids that
  were never seen.
- :func:`nuclear_cytosolic_ratio`: the KTR biosensor readout, from a
  nuclear and a cell segmentation and one intensity image.

    regions = context.regions("545")                # or context.tracks("545")
    means = regions.measure(context.raw("545"))     # one value per regions.ids
    pattern = regions.paint(np.clip(duty, 0, 1))    # each object filled with its duty
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import ndimage

_STATS = {
    "mean": ndimage.mean,
    "median": ndimage.median,
    "min": ndimage.minimum,
    "max": ndimage.maximum,
    "sum": ndimage.sum_labels,
    "std": ndimage.standard_deviation,
    "var": ndimage.variance,
}


class Regions:
    """
    Labelled regions of one frame (0 = background). Every per-object result
    is an array aligned with :attr:`ids`.
    """

    def __init__(self, labels: np.ndarray):
        self.labels = np.asarray(labels)
        self._ids: np.ndarray | None = None

    @property
    def ids(self) -> np.ndarray:
        """The object ids present, ascending."""
        if self._ids is None:
            lab = self.labels
            self._ids = np.unique(lab[lab > 0]).astype(np.int64)
        return self._ids

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} objects, labels {self.labels.shape})"

    def mask(self, region_id: int) -> np.ndarray:
        """Boolean mask of one object."""
        return self.labels == region_id

    def areas(self) -> np.ndarray:
        """Pixels per object."""
        ids = self.ids
        if len(ids) == 0:
            return np.zeros(0, dtype=np.int64)
        ones = np.ones(self.labels.shape, dtype=np.int64)
        return np.asarray(ndimage.sum_labels(ones, self.labels, ids), dtype=np.int64)

    def centroids(self) -> np.ndarray:
        """``(N, 2)`` centroids as ``(y, x)`` in pixels."""
        ids = self.ids
        if len(ids) == 0:
            return np.zeros((0, 2), dtype=float)
        ones = np.ones(self.labels.shape, dtype=float)
        return np.asarray(
            ndimage.center_of_mass(ones, self.labels, ids), dtype=float
        ).reshape(-1, 2)

    def measure(self, image: np.ndarray, stat: str = "mean") -> np.ndarray:
        """
        One statistic of ``image`` per object: ``"mean"`` (default),
        ``"median"``, ``"min"``, ``"max"``, ``"sum"``, ``"std"`` or ``"var"``.
        """
        fn = _STATS.get(stat)
        if fn is None:
            raise ValueError(f"unknown stat {stat!r}; choose from {sorted(_STATS)}")
        image = np.asarray(image)
        if image.shape != self.labels.shape:
            raise ValueError(
                f"image shape {image.shape} does not match labels {self.labels.shape}"
            )
        ids = self.ids
        if len(ids) == 0:
            return np.zeros(0, dtype=float)
        return np.asarray(fn(image, self.labels, ids), dtype=float)

    def paint(
        self,
        values,
        ids=None,
        background: float = 0.0,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        An image with every object filled with its value. ``values`` is a
        scalar (every object), a ``{id: value}`` dict, or an array aligned
        with :attr:`ids` (or with ``ids`` if given). Pixels outside the
        objects, and objects without a value, get ``background``.
        """
        size = int(self.labels.max()) + 1 if self.labels.size else 1
        lut = np.full(size, background, dtype=dtype)
        if isinstance(values, dict):
            for k, v in values.items():
                if 0 < int(k) < size:
                    lut[int(k)] = v
        elif np.ndim(values) == 0:
            lut[self.ids] = values
        else:
            target = self.ids if ids is None else np.asarray(ids, dtype=np.int64)
            values = np.asarray(values)
            if values.shape != target.shape:
                raise ValueError(
                    f"{len(values)} values for {len(target)} ids; pass one value per id"
                )
            keep = (target > 0) & (target < size)
            lut[target[keep]] = values[keep]
        return lut[self.labels]

    def select(self, ids) -> Regions:
        """The same frame keeping only the objects in ``ids``."""
        keep = np.isin(self.labels, np.asarray(list(ids), dtype=np.int64))
        return Regions(np.where(keep, self.labels, 0))

    def owner_of(self, other: Regions) -> np.ndarray:
        """
        For each object of ``other`` (say, a nucleus), the id of the object
        of this frame (the cell) under its centroid, or 0 if there is none.
        Aligned with ``other.ids``.
        """
        if other.labels.shape != self.labels.shape:
            raise ValueError("owner_of needs two label images of the same shape")
        c = other.centroids()
        if len(c) == 0:
            return np.zeros(0, dtype=np.int64)
        yy = np.clip(np.rint(c[:, 0]).astype(int), 0, self.labels.shape[0] - 1)
        xx = np.clip(np.rint(c[:, 1]).astype(int), 0, self.labels.shape[1] - 1)
        return self.labels[yy, xx].astype(np.int64)


def as_regions(labels) -> Regions:
    return labels if isinstance(labels, Regions) else Regions(labels)


class PerTrack:
    """
    A value remembered per track id (or region id). ``update`` stores the
    latest value for each id; ``get`` reads values back aligned with the ids
    you pass, filling ids never seen with a default.
    """

    def __init__(self, values: dict | None = None):
        self._values: dict[int, float] = {
            int(k): float(v) for k, v in (values or {}).items()
        }

    def update(self, ids, values) -> None:
        for i, v in zip(ids, values, strict=True):
            self._values[int(i)] = float(v)

    def get(self, ids, default=None) -> np.ndarray:
        """
        Values for ``ids``. An unknown id gets ``default``: a number,
        ``"median"`` or ``"mean"`` of the values known so far, or NaN when
        ``default`` is None.
        """
        fill = self._fill(default)
        return np.array([self._values.get(int(i), fill) for i in ids], dtype=float)

    def known(self, ids) -> np.ndarray:
        """Boolean array: which of ``ids`` have a value."""
        return np.array([int(i) in self._values for i in ids], dtype=bool)

    def _fill(self, default) -> float:
        if default is None:
            return float("nan")
        if isinstance(default, str):
            if not self._values:
                return float("nan")
            values = np.fromiter(self._values.values(), dtype=float)
            if default == "median":
                return float(np.median(values))
            if default == "mean":
                return float(np.mean(values))
            raise ValueError(
                f"unknown default {default!r}; use a number, 'median' or 'mean'"
            )
        return float(default)

    @property
    def ids(self) -> np.ndarray:
        return np.array(sorted(self._values), dtype=np.int64)

    def values(self) -> np.ndarray:
        return np.array([self._values[i] for i in sorted(self._values)], dtype=float)

    def forget(self, ids) -> None:
        for i in ids:
            self._values.pop(int(i), None)

    def clear(self) -> None:
        self._values.clear()

    def as_dict(self) -> dict[int, float]:
        return dict(self._values)

    def __contains__(self, track_id) -> bool:
        return int(track_id) in self._values

    def __getitem__(self, track_id) -> float:
        return self._values[int(track_id)]

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"PerTrack({len(self._values)} ids)"


class NuclearCytosolicRatio(NamedTuple):
    """Per nucleus: its id, the cell it sits in (0 if none), and the readout."""

    nucleus_ids: np.ndarray
    cell_ids: np.ndarray
    nuclear: np.ndarray
    cytosolic: np.ndarray
    ratio: np.ndarray


def nuclear_cytosolic_ratio(
    nuclei, cells, image: np.ndarray, stat: str = "mean"
) -> NuclearCytosolicRatio:
    """
    The KTR readout: for every nucleus, its intensity in ``image`` divided by
    the intensity of the cytosol of the cell it sits in (that cell minus
    every nucleus). ``nuclei`` and ``cells`` are label images or
    :class:`Regions` of the same frame. Nuclei outside any cell, or in a
    cell with no cytosol pixels, get NaN.
    """
    nuclei = as_regions(nuclei)
    cells = as_regions(cells)
    owner = cells.owner_of(nuclei)
    nuclear = nuclei.measure(image, stat)
    cytosol = Regions(np.where(nuclei.labels > 0, 0, cells.labels))
    by_cell = dict(zip(cytosol.ids.tolist(), cytosol.measure(image, stat), strict=True))
    cytosolic = np.array([by_cell.get(int(c), np.nan) for c in owner], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = nuclear / cytosolic
    ratio = np.where((owner > 0) & (cytosolic > 0), ratio, np.nan)
    return NuclearCytosolicRatio(nuclei.ids, owner, nuclear, cytosolic, ratio)
