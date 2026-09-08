"""
Two pattern methods built on tracking, included verbatim in the user
documentation (documentation/tracking.md) and exercised by
tests/test_doc_examples.py.

Register them with::

    run_pyclm(
        experiment_dir,
        pattern_methods={
            "leader_cells": LeaderCells,
            "intensity_program": IntensityProgram,
        },
    )
"""

import numpy as np

from pyclm import PatternMethod, PerTrack
from pyclm.core.patterns.fbc_cell_movement import RotateCcwModel


class LeaderCells(RotateCcwModel):
    """
    On the first pattern of the run, a ``fraction`` of the cells are chosen
    as leaders. Leaders receive the counter-clockwise half-cell stimulus of
    ``rotate_ccw`` for the rest of the movie; every other cell receives
    nothing.
    """

    name = "leader_cells"

    def __init__(self, channel="545", fraction=0.2, seed=0, **kwargs):
        # tracks=True makes the per-cell loop run on tracked labels, so the
        # region label seen by process_prop is a track id that persists
        super().__init__(channel=channel, tracks=True, **kwargs)
        self.fraction = float(np.clip(fraction, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.leaders: set[int] | None = None

    def cell_labels(self, context) -> np.ndarray:
        tracks = context.tracks(self.channel)

        if self.leaders is None and len(tracks) > 0:
            # first frame with cells: draw the leaders once (at least one
            # unless fraction is 0)
            n = round(len(tracks) * self.fraction)
            n = max(1, n) if self.fraction > 0 else 0
            chosen = self.rng.choice(tracks.ids, size=n, replace=False)
            self.leaders = {int(i) for i in chosen}

        # keep only the leaders; rotate_ccw's per-cell loop does the rest
        return tracks.select(self.leaders or ()).labels


class IntensityProgram(PatternMethod):
    """
    A three-phase programme for every cell: no light for ``dark_min``
    minutes, full light for ``light_min`` minutes, then feedback control to
    the midpoint between the intensity each cell showed at the end of the
    dark phase and at the end of the light phase.

    The controller is proportional: a cell at its target gets a 50 % duty
    pattern, a cell far below it full light, a cell far above it none. A
    large ``gain`` makes it bang-bang. Cells first seen after the light
    phase have no calibration of their own and use the population medians.
    """

    name = "intensity_program"

    def __init__(
        self, channel="545", dark_min=30.0, light_min=30.0, gain=2.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.channel = channel
        # the raw frame gives the intensities, the tracks give each cell a lasting id
        self.add_requirement(channel, raw=True, tracks=True)
        self.dark_s = float(dark_min) * 60.0
        self.light_s = float(light_min) * 60.0
        self.gain = float(gain)
        # per track id: the intensity at the end of the dark and light phases
        self.low = PerTrack()
        self.high = PerTrack()

    def generate(self, context) -> np.ndarray:
        tracks = context.tracks(self.channel)
        means = tracks.measure(context.raw(self.channel))  # one value per tracks.ids

        if context.time < self.dark_s:
            # phase 1: dark. Keep overwriting, so the last value is the settled one.
            self.low.update(tracks.ids, means)
            return tracks.paint(0.0)

        if context.time < self.dark_s + self.light_s:
            # phase 2: every cell fully lit
            self.high.update(tracks.ids, means)
            return tracks.paint(1.0)

        # phase 3: hold each cell at the midpoint of its own calibration;
        # cells without one use the population median
        low = self.low.get(tracks.ids, default="median")
        high = self.high.get(tracks.ids, default="median")
        target = 0.5 * (low + high)
        duty = 0.5 + self.gain * (target - means) / np.maximum(high - low, 1e-6)
        return tracks.paint(np.clip(np.nan_to_num(duty, nan=0.0), 0.0, 1.0))
