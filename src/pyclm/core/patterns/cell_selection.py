"""
Closed-loop methods that light whole cells, chosen by a rule: every cell
(``all_cells``), a seeded random fraction kept per track (``random_cells``),
or cells chosen by a property (``select_cells``, which also lights a band
at the colony edge). Unlike the per-cell direction family, a chosen cell is
lit entirely, so they are also the controls for it: same cells, no polarity.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..measure import Regions
from .pattern import PatternMethod
from .zoo import ZooMeta


class CellSelectionMethod(PatternMethod):
    """Base: light the segmented objects of ``channel`` that :meth:`chosen` returns."""

    name = "cell_selection_base"
    needs_tracks = False

    def __init__(self, channel=None, **kwargs):
        super().__init__(**kwargs)
        if channel is None:
            raise AttributeError(
                f"{self.name} must be given the channel whose cells it lights "
                'in the .toml: e.g. channel = "638"'
            )
        self.channel = channel
        self.add_requirement(
            channel, seg=not self.needs_tracks, tracks=self.needs_tracks
        )

    def labels(self, context) -> np.ndarray:
        if self.needs_tracks:
            tracks = context.tracks(self.channel)
            if tracks is None:
                return np.zeros(tuple(int(v) for v in self.pattern_shape), np.int32)
            return np.asarray(tracks.labels)
        return np.asarray(context.segmentation(self.channel))

    def chosen(self, context, regions: Regions) -> np.ndarray:
        """The ids of the cells to light."""
        return regions.ids

    def generate(self, context):
        regions = Regions(self.labels(context))
        ids = self.chosen(context, regions)
        return np.isin(regions.labels, ids) & (regions.labels > 0)


def _grow(mask: np.ndarray, distance_px: float) -> np.ndarray:
    """``mask`` grown by ``distance_px`` (shrunk for a negative one)."""
    if distance_px > 0:
        return ndimage.distance_transform_edt(~mask) <= distance_px
    if distance_px < 0:
        return ndimage.distance_transform_edt(mask) > -distance_px
    return mask


class AllCellsPattern(CellSelectionMethod):
    """
    Lights every segmented cell of ``channel``; ``dilate_um`` grows (or,
    negative, shrinks) the lit area around the cells, and ``invert`` lights
    everything but them (the background).
    """

    name = "all_cells"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"channel": "638", "dilate_um": 2},
        title="All Cells",
        description="Every segmented cell lit (or, with invert, everything but the cells).",
        group="per_cell",
    )

    def __init__(self, channel=None, invert=False, dilate_um=0.0, **kwargs):
        """
        :param channel: the channel whose cells are lit
        :param invert: light the background instead of the cells
        :param dilate_um: grow the lit cells by this distance (negative: shrink), in um
        """
        super().__init__(channel=channel, **kwargs)
        self.invert = bool(invert)
        self.dilate_um = float(dilate_um)

    def generate(self, context):
        mask = np.asarray(super().generate(context), bool)
        mask = _grow(mask, self.dilate_um / self.pixel_size_um)
        if self.invert:
            mask = ~mask
        return mask.astype(np.float16)


class RandomCellsPattern(CellSelectionMethod):
    """
    Lights a random ``fraction`` of the tracked cells of ``channel``. Each
    track is drawn once, from ``seed`` and its id, so a cell keeps its
    choice for the whole run (and in a rerun with the same seed); cells that
    appear later are drawn as they appear. Needs a ``[tracking]`` table.
    """

    name = "random_cells"
    needs_tracks = True
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"channel": "638", "fraction": 0.3, "seed": 1},
        title="Random Cells",
        description="A seeded random fraction of the tracked cells lit; each cell keeps its draw for the run.",
        group="per_cell",
    )

    def __init__(self, channel=None, fraction=0.5, seed=0, **kwargs):
        """
        :param channel: the channel whose tracked cells are lit
        :param fraction: the fraction of cells lit (0-1)
        :param seed: the random seed
        """
        super().__init__(channel=channel, **kwargs)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{self.name}: fraction must be in 0-1, got {fraction}")
        self.fraction = float(fraction)
        self.seed = int(seed)

    def draw(self, track_id: int) -> float:
        """The track's uniform draw in [0, 1): it is lit when below ``fraction``."""
        return float(np.random.default_rng([self.seed, int(track_id)]).random())

    def chosen(self, context, regions):
        return np.array(
            [i for i in regions.ids if self.draw(i) < self.fraction], dtype=np.int64
        )


def _tissue(labels: np.ndarray, close_px: int = 3) -> np.ndarray:
    """The colony: the cells' union with the gaps between them closed and holes filled."""
    mask = labels > 0
    if close_px > 0:
        structure = ndimage.generate_binary_structure(2, 1)
        mask = ndimage.binary_closing(mask, structure, iterations=close_px)
    return ndimage.binary_fill_holes(mask)


class SelectCellsPattern(CellSelectionMethod):
    """
    Lights the cells of ``channel`` chosen by one property, ``by``:

    - ``"area"``: the cell's area, in um²;
    - ``"intensity"``: its mean intensity in ``channel``, in camera counts;
    - ``"distance_center"``: its centroid's distance from the field center, in um;
    - ``"distance_edge"``: its centroid's distance from the colony edge, in um
      (the colony is the cells with the gaps between them closed).

    A cell is lit when its value is in ``[min, max]`` (either may be left
    out); ``top_n`` then keeps the ``top_n`` cells with the largest values
    (a negative ``top_n``, the smallest: ``top_n = -1`` with
    ``by = "distance_center"`` is the cell nearest the center).

    ``band_um`` lights pixels instead of cells: every pixel of the colony
    within ``band_um`` of its edge (with ``by = "distance_edge"`` only).
    """

    name = "select_cells"
    BY = ("area", "intensity", "distance_center", "distance_edge")
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"channel": "638", "by": "distance_center", "max": 150},
        title="Select Cells",
        description="Cells chosen by area, intensity, or distance from the center or the colony edge; or a band at the edge.",
        group="per_cell",
    )

    def __init__(
        self,
        channel=None,
        by=None,
        min=None,
        max=None,
        top_n=None,
        band_um=None,
        **kwargs,
    ):
        """
        :param channel: the channel whose cells are chosen and lit
        :param by: "area", "intensity", "distance_center" or "distance_edge"
        :param min: lowest value lit (in the unit of ``by``)
        :param max: highest value lit
        :param top_n: keep the n cells with the largest values (negative: smallest)
        :param band_um: light the colony within this distance of its edge, in um
        """
        super().__init__(channel=channel, **kwargs)
        if by not in self.BY:
            raise ValueError(
                f"{self.name}: by must be one of {', '.join(map(repr, self.BY))}, "
                f"got {by!r}"
            )
        if band_um is not None:
            if by != "distance_edge":
                raise ValueError(
                    f"{self.name}: band_um lights the colony edge; it needs "
                    'by = "distance_edge"'
                )
            if min is not None or max is not None or top_n is not None:
                raise ValueError(
                    f"{self.name}: band_um lights pixels, not cells; leave out "
                    "min, max and top_n"
                )
            if band_um <= 0:
                raise ValueError(f"{self.name}: band_um must be positive")
        elif min is None and max is None and top_n is None:
            raise ValueError(
                f"{self.name}: give min, max or top_n (or band_um); "
                'to light every cell use method = "all_cells"'
            )
        if min is not None and max is not None and min > max:
            raise ValueError(f"{self.name}: min ({min}) is above max ({max})")
        if top_n is not None and int(top_n) == 0:
            raise ValueError(f"{self.name}: top_n must not be 0")
        self.by = by
        self.min = min
        self.max = max
        self.top_n = None if top_n is None else int(top_n)
        self.band_um = None if band_um is None else float(band_um)
        if by == "intensity":
            self.add_requirement(channel, raw=True)

    def values(self, context, regions: Regions) -> np.ndarray:
        """Every cell's value of ``by``, aligned with ``regions.ids``."""
        px = self.pixel_size_um
        if self.by == "area":
            return regions.areas() * px**2
        if self.by == "intensity":
            return regions.measure(context.raw(self.channel))
        centroids = regions.centroids()
        if self.by == "distance_center":
            h, w = self.pattern_shape
            center = np.array([(h - 1) / 2, (w - 1) / 2])
            return np.hypot(*(centroids - center).T) * px
        depth = ndimage.distance_transform_edt(_tissue(regions.labels))
        yy = np.clip(np.rint(centroids[:, 0]).astype(int), 0, depth.shape[0] - 1)
        xx = np.clip(np.rint(centroids[:, 1]).astype(int), 0, depth.shape[1] - 1)
        return depth[yy, xx] * px

    def chosen(self, context, regions):
        ids = regions.ids
        if len(ids) == 0:
            return ids
        values = self.values(context, regions)
        keep = np.ones(len(ids), bool)
        if self.min is not None:
            keep &= values >= self.min
        if self.max is not None:
            keep &= values <= self.max
        ids, values = ids[keep], values[keep]
        if self.top_n is not None:
            order = np.argsort(values, kind="stable")
            if self.top_n > 0:
                order = order[::-1]
            ids = ids[order[: abs(self.top_n)]]
        return ids

    def generate(self, context):
        if self.band_um is None:
            return np.asarray(super().generate(context)).astype(np.float16)
        tissue = _tissue(self.labels(context))
        depth = ndimage.distance_transform_edt(tissue) * self.pixel_size_um
        return (tissue & (depth <= self.band_um)).astype(np.float16)
