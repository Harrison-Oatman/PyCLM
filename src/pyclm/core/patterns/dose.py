"""
Graded light doses: one helper shared by the methods that light a region at
a fraction of full light (``dose_grid``, ``nucleus_clamp`` with a ``p`` or
``pi`` controller, ``population_clamp``).

``dose_mode`` picks how a dose between 0 and 1 reaches the sample:

- ``"gray"`` (default): the DMD shows the dose as a gray level.
- ``"dither"``: every pixel is fully on or off; a fixed 8 x 8 ordered
  (Bayer) dither lights a ``dose`` fraction of the pixels of a region.
- ``"duty"``: every pixel is fully on or off; a region is lit for a
  ``dose`` fraction of every ``period_min`` (lit first, then dark).
"""

from __future__ import annotations

import numpy as np

DOSE_MODES = ("gray", "dither", "duty")

# the classic recursive Bayer matrix, thresholds in (0, 1)
_BAYER2 = np.array([[0, 2], [3, 1]])
_BAYER4 = np.block([[4 * _BAYER2, 4 * _BAYER2 + 2], [4 * _BAYER2 + 3, 4 * _BAYER2 + 1]])
_BAYER8 = np.block([[4 * _BAYER4, 4 * _BAYER4 + 2], [4 * _BAYER4 + 3, 4 * _BAYER4 + 1]])
BAYER = (_BAYER8 + 0.5) / 64.0


def check_dose_mode(name: str, dose_mode: str, period_min: float) -> None:
    """Raise a ValueError naming the method for a bad ``dose_mode`` or ``period_min``."""
    if dose_mode not in DOSE_MODES:
        raise ValueError(
            f"{name}: dose_mode must be one of {', '.join(map(repr, DOSE_MODES))}, "
            f"got {dose_mode!r}"
        )
    if period_min is None or period_min <= 0:
        raise ValueError(f"{name}: period_min must be positive, got {period_min}")


def dither(dose: np.ndarray) -> np.ndarray:
    """Ordered dither: a pixel is on where its dose exceeds the Bayer threshold."""
    dose = np.asarray(dose, dtype=np.float32)
    h, w = dose.shape
    reps = (-(-h // 8), -(-w // 8))
    threshold = np.tile(BAYER, reps)[:h, :w]
    return (dose > threshold).astype(np.float32)


def duty(dose: np.ndarray, t_minutes: float, period_min: float) -> np.ndarray:
    """Temporal duty: a pixel is on for the first ``dose`` fraction of each period."""
    phase = (float(t_minutes) % period_min) / period_min
    return (np.asarray(dose, dtype=np.float32) > phase).astype(np.float32)


def deliver(
    dose, dose_mode: str = "gray", t_minutes: float = 0.0, period_min: float = 10.0
) -> np.ndarray:
    """A dose image (0-1) as the pattern to project, by ``dose_mode``."""
    dose = np.clip(np.asarray(dose, dtype=np.float32), 0.0, 1.0)
    if dose_mode == "gray":
        return dose
    if dose_mode == "dither":
        return dither(dose)
    if dose_mode == "duty":
        return duty(dose, t_minutes, period_min)
    raise ValueError(f"unknown dose_mode {dose_mode!r}")
