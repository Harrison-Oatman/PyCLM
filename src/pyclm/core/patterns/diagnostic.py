"""
Diagnostics: methods that check the setup rather than experiment on cells.

``game_of_life`` plays Conway's Game of Life on a board of squares, read
back from the camera each step: on a fluorescent slide, the board only
evolves correctly when what the DMD projects lands where the camera expects
it, so it is a quick, visual check of the camera → DMD calibration (and of
the closed loop as a whole). ``agreement`` says how much of the board was
read back as projected.
"""

from __future__ import annotations

import logging

import numpy as np

from .pattern import PatternMethod
from .zoo import ZooMeta

logger = logging.getLogger(__name__)


def life_step(board: np.ndarray) -> np.ndarray:
    """One generation of Conway's rules on a board that wraps at its edges."""
    board = np.asarray(board, bool)
    neighbors = sum(
        np.roll(np.roll(board, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    return (neighbors == 3) | (board & (neighbors == 2))


class GameOfLifePattern(PatternMethod):
    """
    A ``cells`` x ``cells`` board of squares over the field, each square
    ``fill`` of its pitch wide. The first board is random (``seed``); after
    that each step reads the board back from the imaged stimulation frame (a
    square is alive when its mean is above ``threshold``, in camera counts,
    and brighter than the border around it), applies Conway's rules and
    projects the next board. ``agreement`` is the fraction of squares read
    back as they were projected; a calibration that is off shows as a low
    agreement and a board that falls apart.
    """

    name = "game_of_life"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"cells": 30, "fill": 0.7, "threshold": 2000, "seed": 0},
        title="Game of Life",
        description="Conway's Game of Life read back from the camera each step: a visual check of the DMD calibration on a fluorescent slide.",
        pending=(
            "frames of a fluorescent slide under the projected boards "
            "(the board is read back from the camera each step)"
        ),
        group="diagnostic",
    )

    def __init__(self, cells=50, fill=0.7, threshold=None, seed=0, **kwargs):
        """
        :param cells: squares along each side of the board
        :param fill: width of each square as a fraction of its pitch (0-1)
        :param threshold: mean intensity (camera counts) of a lit square
        :param seed: the random seed of the first board
        """
        super().__init__(**kwargs)
        if int(cells) < 3:
            raise ValueError(f"{self.name}: cells must be at least 3, got {cells}")
        if not 0 < fill <= 1:
            raise ValueError(f"{self.name}: fill must be in (0, 1], got {fill}")
        if threshold is None:
            raise ValueError(
                f"{self.name}: threshold (the camera counts of a lit square on "
                "your slide) is required"
            )
        self.cells = int(cells)
        self.fill = float(fill)
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.board = np.random.default_rng(self.seed).random((self.cells,) * 2) < 0.5
        self.agreement: float | None = None
        self._projected = False
        self.request_stim(raw=True)

    # ------------------------------------------------------------ board <-> image
    def _squares(self, shape):
        """Each pixel's square ``(row, col)`` and whether it is inside the lit part."""
        h, w = shape
        yy, xx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        pitch_y, pitch_x = h / self.cells, w / self.cells
        row = np.minimum((yy // pitch_y).astype(int), self.cells - 1)
        col = np.minimum((xx // pitch_x).astype(int), self.cells - 1)
        fy = (yy - row * pitch_y) / pitch_y
        fx = (xx - col * pitch_x) / pitch_x
        edge = (1 - self.fill) / 2
        inside = (fy >= edge) & (fy < 1 - edge) & (fx >= edge) & (fx < 1 - edge)
        return row, col, inside

    def draw(self, board: np.ndarray, shape=None) -> np.ndarray:
        """The pattern of a board."""
        shape = tuple(int(v) for v in (shape or self.pattern_shape))
        row, col, inside = self._squares(shape)
        return (inside & board[row, col]).astype(np.float32)

    def read(self, image: np.ndarray) -> np.ndarray:
        """The board seen in a camera frame."""
        image = np.asarray(image, dtype=float)
        row, col, inside = self._squares(image.shape)
        flat = row * self.cells + col
        n = self.cells * self.cells
        inner_sum = np.bincount(flat[inside], image[inside], minlength=n)
        inner_n = np.bincount(flat[inside], minlength=n)
        border_sum = np.bincount(flat[~inside], image[~inside], minlength=n)
        border_n = np.bincount(flat[~inside], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            inner = inner_sum / inner_n
            border = np.where(border_n > 0, border_sum / np.maximum(border_n, 1), 0)
        alive = (inner > self.threshold) & (inner > border)
        return alive.reshape(self.cells, self.cells)

    # ------------------------------------------------------------ the pattern
    def generate(self, context):
        if self._projected:
            frame = context.stim_raw()
            if frame is not None:
                seen = self.read(frame)
                self.agreement = float(np.mean(seen == self.board))
                logger.info(
                    f"{self.name}: {self.agreement:.1%} of squares read back as "
                    f"projected ({int(seen.sum())} alive seen, "
                    f"{int(self.board.sum())} projected)"
                )
                self.board = life_step(seen)
        self._projected = True
        return self.draw(self.board)
