"""
Grid acquisition: a position made of tiles, laid out by MicroManager's tile
creator, imaged as one stitched frame.

MicroManager's Create Grid writes one position per tile, labelled
``<prefix>-<n>-<col>_<row>`` with ``GridRow`` / ``GridCol`` set, spaced by
the camera field (through the pixel-size affine, so tiles placed by row and
column abut). :func:`group_tiles` folds those entries into one
:class:`~pyclm.core.experiments.MicroscopePosition` per grid that carries
the tiles; :func:`geometry_for` turns it, once the camera ROI and pixel size
are known, into a :class:`GridGeometry`; :func:`stitch` and :func:`cut`
move frames and patterns between the tiles and the stitched frame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

import numpy as np

from .experiments import MicroscopePosition

# MicroManager's TileCreator: labelPrefix + "-" + "%03d" % col + "_" + "%03d" % row
TILE_LABEL = re.compile(r"^(?P<prefix>.+)-(?P<col>\d{3})_(?P<row>\d{3})$")


@dataclass(frozen=True)
class GridGeometry:
    """
    How a grid's tiles sit in the stitched frame, in unbinned camera pixels.

    ``pitch_x_px`` / ``pitch_y_px`` are the distances between column / row
    neighbours; ``tile_w`` / ``tile_h`` the camera ROI; ``tiles`` the
    ``(row, col)`` of each tile in acquisition order.
    """

    rows: int
    columns: int
    pitch_x_px: float
    pitch_y_px: float
    tile_h: int
    tile_w: int
    tiles: tuple[tuple[int, int], ...]

    def pitch(self, binning: int = 1) -> tuple[int, int]:
        """(row pitch, column pitch) in pixels at ``binning``."""
        return (
            round(self.pitch_y_px / binning),
            round(self.pitch_x_px / binning),
        )

    def tile_shape(self, binning: int = 1) -> tuple[int, int]:
        return (self.tile_h // binning, self.tile_w // binning)

    def shape(self, binning: int = 1) -> tuple[int, int]:
        """The stitched frame's (height, width) at ``binning``."""
        h, w = self.tile_shape(binning)
        py, px = self.pitch(binning)
        return ((self.rows - 1) * py + h, (self.columns - 1) * px + w)

    def overlap(self, binning: int = 1) -> tuple[int, int]:
        """(rows, columns) overlap in pixels between neighbours; negative is a gap."""
        h, w = self.tile_shape(binning)
        py, px = self.pitch(binning)
        return (h - py, w - px)

    def tile_slices(self, row: int, col: int, binning: int = 1) -> tuple[slice, slice]:
        h, w = self.tile_shape(binning)
        py, px = self.pitch(binning)
        y0, x0 = row * py, col * px
        return (slice(y0, y0 + h), slice(x0, x0 + w))

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "pitch_px": [self.pitch_x_px, self.pitch_y_px],
            "tile_shape": [self.tile_h, self.tile_w],
            "tiles": [list(rc) for rc in self.tiles],
        }


# ------------------------------------------------------------- from a position list
def group_tiles(
    positions: list[MicroscopePosition],
    raw_labels: list[str],
    grid_rc: list[tuple[int, int] | None],
) -> list[MicroscopePosition]:
    """
    Fold the tiles of MicroManager grids into one position per grid.

    ``raw_labels`` are the labels as written by MicroManager (before ``-``
    became ``.``); ``grid_rc`` the ``(GridRow, GridCol)`` of each entry, or
    None. A grid is every entry whose label matches :data:`TILE_LABEL` with
    the same prefix; the grid position is labelled by the prefix (``-`` →
    ``.``), sits at the centre of its tiles, and carries them in
    ``tiles`` (each with ``grid_rc``) with the pitch in µm in ``grid``.
    Entries that are not tiles pass through unchanged, in order.
    """
    out: list[MicroscopePosition] = []
    grids: dict[str, list[tuple[MicroscopePosition, int, int]]] = {}
    order: list[str] = []
    for pos, raw, rc in zip(positions, raw_labels, grid_rc, strict=True):
        m = TILE_LABEL.match(raw or "")
        if m is None:
            out.append(pos)
            continue
        col, row = int(m.group("col")), int(m.group("row"))
        if rc is not None and rc != (0, 0) and rc != (row, col):
            raise ValueError(
                f"position '{raw}': GridRow/GridCol {rc} disagree with the label "
                f"(row {row}, column {col})"
            )
        prefix = m.group("prefix")
        if prefix not in grids:
            grids[prefix] = []
            order.append(prefix)
            out.append(None)  # type: ignore[arg-type]  (placeholder keeps the order)
        grids[prefix].append((pos, row, col))

    for prefix in order:
        entries = grids[prefix]
        grid = _fold(prefix, entries)
        out[out.index(None)] = grid
    return out


def _fold(prefix: str, entries: list[tuple[MicroscopePosition, int, int]]):
    label = prefix.replace("-", ".")
    rcs = [(r, c) for _, r, c in entries]
    if len(set(rcs)) != len(rcs):
        raise ValueError(f"grid '{label}': duplicate tiles {rcs}")
    rows = max(r for r, _ in rcs) + 1
    cols = max(c for _, c in rcs) + 1
    if rows * cols != len(rcs) or min(r for r, _ in rcs) or min(c for _, c in rcs):
        raise ValueError(
            f"grid '{label}': {len(rcs)} tiles do not form a complete "
            f"{rows} x {cols} rectangle (tiles must not be deleted from a grid)"
        )
    by_rc = {(r, c): p for p, r, c in entries}

    def pitch(pairs):
        dists = [float(np.hypot(b.x - a.x, b.y - a.y)) for a, b in pairs]
        if not dists:
            return 0.0
        p = median(dists)
        if p > 0 and max(abs(d - p) for d in dists) > 0.02 * p:
            raise ValueError(
                f"grid '{label}': tile spacing is not uniform ({min(dists):.2f} to "
                f"{max(dists):.2f} um); was the grid edited by hand?"
            )
        return p

    col_pitch = pitch(
        [
            (by_rc[(r, c)], by_rc[(r, c + 1)])
            for r in range(rows)
            for c in range(cols - 1)
        ]
    )
    row_pitch = pitch(
        [
            (by_rc[(r, c)], by_rc[(r + 1, c)])
            for r in range(rows - 1)
            for c in range(cols)
        ]
    )

    tiles = []
    for pos, r, c in entries:
        tile = MicroscopePosition(
            pos.x,
            pos.y,
            pos.z,
            label=f"{label}.{c:03d}_{r:03d}",
            extras=dict(pos.extras),
        )
        tile.grid_rc = (r, c)
        tiles.append(tile)

    n = len(tiles)
    extras: dict = {}
    for key in tiles[0].extras:
        values = [t.extras.get(key) for t in tiles]
        if all(isinstance(v, int | float) for v in values):
            extras[key] = float(sum(values)) / n
    centre = MicroscopePosition(
        sum(t.x for t in tiles) / n,
        sum(t.y for t in tiles) / n,
        sum(t.z for t in tiles) / n,
        label=label,
        extras=extras,
    )
    centre.tiles = tiles
    centre.grid = {"rows": rows, "columns": cols, "pitch_um": [col_pitch, row_pitch]}
    return centre


# ------------------------------------------------------------- at run time
def geometry_for(
    position: MicroscopePosition, roi, pixel_size_um: float
) -> GridGeometry | None:
    """
    The :class:`GridGeometry` of a grid position given the camera ROI
    ``(x, y, width, height)`` (unbinned) and the unbinned pixel size; None
    for a plain position.
    """
    tiles = getattr(position, "tiles", None)
    if not tiles:
        return None
    info = position.grid
    col_um, row_um = info["pitch_um"]
    px = float(pixel_size_um)
    if px <= 0:
        raise ValueError("pixel size must be positive to lay out a grid")
    return GridGeometry(
        rows=int(info["rows"]),
        columns=int(info["columns"]),
        pitch_x_px=col_um / px,
        pitch_y_px=row_um / px,
        tile_h=int(roi[3]),
        tile_w=int(roi[2]),
        tiles=tuple(tuple(t.grid_rc) for t in tiles),
    )


def stitch(frames, geometry: GridGeometry, binning: int = 1) -> np.ndarray:
    """
    Place the tiles' frames (in ``geometry.tiles`` order) into one stitched
    frame at ``binning``. Placement only: where tiles overlap the later one
    wins; where the pitch exceeds the tile the gap stays zero. Frames smaller
    than the camera ROI says (the virtual microscope's TIFs stand for a
    larger camera) are laid out at their own scale.
    """
    frames = [np.asarray(f) for f in frames]
    if len(frames) != len(geometry.tiles):
        raise ValueError(f"{len(frames)} frames for {len(geometry.tiles)} tiles")
    h, w = geometry.tile_shape(binning)
    fh, fw = frames[0].shape[:2]
    scale = float(binning)
    if (fh, fw) != (h, w) and fh > 0 and fw > 0 and abs(h / fh - w / fw) < 1e-6:
        scale = binning * (h / fh)
        h, w = fh, fw
    py = round(geometry.pitch_y_px / scale)
    px = round(geometry.pitch_x_px / scale)
    shape = ((geometry.rows - 1) * py + h, (geometry.columns - 1) * px + w)
    canvas = np.zeros((*shape, *frames[0].shape[2:]), dtype=frames[0].dtype)
    for frame, (row, col) in zip(frames, geometry.tiles, strict=True):
        y0, x0 = row * py, col * px
        th, tw = min(h, frame.shape[0]), min(w, frame.shape[1])
        canvas[y0 : y0 + th, x0 : x0 + tw] = frame[:th, :tw]
    return canvas


def cut(pattern, geometry: GridGeometry, binning: int = 1) -> list[np.ndarray]:
    """
    The tiles' share of a stitched pattern at ``binning``, in
    ``geometry.tiles`` order, each the full tile extent (zero-padded where the
    pattern is smaller than the stitched frame).
    """
    pattern = np.asarray(pattern)
    h, w = geometry.tile_shape(binning)
    out = []
    for row, col in geometry.tiles:
        sy, sx = geometry.tile_slices(row, col, binning)
        tile = np.zeros((h, w), dtype=pattern.dtype)
        part = pattern[sy, sx]
        tile[: part.shape[0], : part.shape[1]] = part
        out.append(tile)
    return out
