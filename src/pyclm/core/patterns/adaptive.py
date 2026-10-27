"""
Adaptive acquisition: wrappers that pass their inner method's pattern
through and change the run itself.

- ``recenter`` moves the stage so the segmented cells stay centered;
- ``drift_correct`` moves it to undo drift measured by registering frames;
- ``auto_exposure`` keeps a channel's bright pixels at a set fraction of
  the camera's range by changing its exposure;
- ``settings_schedule`` changes an exposure, a preset or a device property
  on a list of ``[start_min, value]`` steps.

Each wraps an optional ``[pattern.inner]`` method (none: the field stays
dark). A change requested at one timepoint applies from the next one, the
same delay as the pattern, and is recorded in ``events.parquet``. A stage
move is given to the tracker, so track ids survive it, and the pattern made
before the move is shifted with it, so it still lands on the cells.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from .composite import WrappingPattern
from .zoo import ZooMeta

logger = logging.getLogger(__name__)


def _shift(pattern: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """``pattern`` moved by whole pixels (rows down, columns right), dark where it left."""
    dy, dx = round(dy), round(dx)
    if dy == 0 and dx == 0:
        return pattern
    out = np.zeros_like(pattern)
    h, w = pattern.shape
    if abs(dy) >= h or abs(dx) >= w:
        return out
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = pattern[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


class StageMoverPattern(WrappingPattern):
    """
    Base of the wrappers that move the stage: :meth:`offset_px` says where
    the content is, in pattern pixels from where it should be; the base
    moves the stage to bring it back, within ``deadband_um`` and
    ``max_step_um``, and shifts this timepoint's pattern to match.
    """

    name = "stage_mover_base"
    inner_required = False

    def __init__(self, channel=None, deadband_um=50.0, max_step_um=200.0, **kwargs):
        super().__init__(**kwargs)
        if channel is None:
            raise AttributeError(
                f"{self.name} must be given the channel it measures: "
                'e.g. channel = "638"'
            )
        if deadband_um < 0 or max_step_um <= 0:
            raise ValueError(
                f"{self.name}: deadband_um must not be negative and max_step_um "
                "must be positive"
            )
        self.channel = channel
        self.deadband_um = float(deadband_um)
        self.max_step_um = float(max_step_um)
        self.moves = 0
        self._warned_no_position = False

    def offset_px(self, context) -> tuple[float, float] | None:
        """``(dy, dx)``: how far the content sits from where it should be, in pattern pixels."""
        raise NotImplementedError

    def _in_pattern_px(self, dy: float, dx: float, shape) -> tuple[float, float]:
        """An offset measured on an image of ``shape`` in pattern pixels (binnings may differ)."""
        h, w = self.pattern_shape
        return dy * h / shape[0], dx * w / shape[1]

    def moved(self, dy_px: float, dx_px: float) -> None:
        """Called after a move of the content by ``(dy_px, dx_px)`` was requested."""

    def move(self, context, dy_px: float, dx_px: float) -> tuple[float, float]:
        """
        Request the stage move that brings content at ``(dy_px, dx_px)`` to
        where it should be; returns the pixel offset actually corrected
        (0, 0 for none).
        """
        sx, sy = self.camera_to_stage(dy_px, dx_px)
        length = float(np.hypot(sx, sy))
        if length <= self.deadband_um or length == 0:
            return 0.0, 0.0
        position = context.position()
        if position is None:
            if not self._warned_no_position:
                logger.warning(f"{self.name}: no stage position known; not moving")
                self._warned_no_position = True
            return 0.0, 0.0
        scale = min(1.0, self.max_step_um / length)
        context.set_position(
            x=float(position["x"]) + sx * scale, y=float(position["y"]) + sy * scale
        )
        self.moves += 1
        self.moved(dy_px * scale, dx_px * scale)
        return dy_px * scale, dx_px * scale

    def generate(self, context):
        pattern = self.inner_pattern(context)
        offset = self.offset_px(context)
        if offset is None:
            return pattern
        dy, dx = self.move(context, *offset)
        # the pattern is shown after the move, which carries the content by
        # (-dy, -dx): carry the pattern with it
        return _shift(pattern, -dy, -dx)


class RecenterPattern(StageMoverPattern):
    """
    Keeps the cells of ``channel`` centered: when the center of the
    segmented area is more than ``deadband_um`` from the field center, the
    stage moves to bring it back, by at most ``max_step_um`` per timepoint.
    For a colony that migrates, or an embryo that slides.
    """

    name = "recenter"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "channel": "638",
            "deadband_um": 50,
            "max_step_um": 200,
            "inner": {"method": "all_cells", "channel": "638"},
        },
        title="Recenter",
        description="Moves the stage to keep the segmented cells centered, around any method.",
        pending="a time series of a migrating colony (the stage follows its center)",
        group="adaptive",
    )

    def __init__(self, channel=None, deadband_um=50.0, max_step_um=200.0, **kwargs):
        """
        :param channel: the channel whose segmentation is kept centered
        :param inner: the wrapped method, a ``[pattern.inner]`` table (optional)
        :param deadband_um: distance from the center tolerated without a move, in um
        :param max_step_um: the largest move per timepoint, in um
        """
        super().__init__(
            channel=channel, deadband_um=deadband_um, max_step_um=max_step_um, **kwargs
        )
        self.add_requirement(channel, seg=True)

    def offset_px(self, context):
        mask = np.asarray(context.segmentation(self.channel)) > 0
        if not mask.any():
            return None
        cy, cx = ndimage.center_of_mass(mask)
        h, w = mask.shape
        return self._in_pattern_px(cy - (h - 1) / 2, cx - (w - 1) / 2, mask.shape)


class DriftCorrectPattern(StageMoverPattern):
    """
    Undoes stage or sample drift: each frame of ``channel`` is registered
    (phase correlation) against the first frame (``reference = "first"``)
    or the previous one (``"previous"``, which follows slow changes of the
    sample better), and when the drift exceeds ``deadband_um`` the stage
    moves back by at most ``max_step_um``.
    """

    name = "drift_correct"
    REFERENCES = ("first", "previous")
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "channel": "638",
            "deadband_um": 2,
            "max_step_um": 50,
            "inner": {"method": "circle", "radius_um": 100},
        },
        title="Drift Correct",
        description="Registers each frame against the first (or the previous) and moves the stage to undo drift.",
        pending="a time series with drift (frames registered against the first)",
        group="adaptive",
    )

    def __init__(
        self,
        channel=None,
        deadband_um=2.0,
        max_step_um=50.0,
        reference="first",
        **kwargs,
    ):
        """
        :param channel: the channel whose frames are registered
        :param inner: the wrapped method, a ``[pattern.inner]`` table (optional)
        :param deadband_um: drift tolerated without a move, in um
        :param max_step_um: the largest move per timepoint, in um
        :param reference: "first" or "previous" frame
        """
        super().__init__(
            channel=channel, deadband_um=deadband_um, max_step_um=max_step_um, **kwargs
        )
        if reference not in self.REFERENCES:
            raise ValueError(
                f"{self.name}: reference must be 'first' or 'previous', got {reference!r}"
            )
        self.reference = reference
        self.add_requirement(channel, raw=True, history=2)
        self._first = None
        # drift not yet corrected, (dy, dx) px, for reference = "previous"
        self._residual = np.zeros(2)

    @staticmethod
    def displacement(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """How far the content of ``frame`` moved from ``reference``, ``(dy, dx)`` px."""
        from skimage.registration import phase_cross_correlation

        shift, _error, _phase = phase_cross_correlation(
            np.asarray(reference, np.float32),
            np.asarray(frame, np.float32),
            upsample_factor=10,
            normalization=None,
        )
        # the shift registers frame onto reference: the content moved the other way
        return -np.asarray(shift, float)

    def offset_px(self, context):
        frames = context.history(self.channel, "raw", 2)
        if not frames:
            return None
        frame = frames[-1]
        if self.reference == "first":
            if self._first is None:
                self._first = np.array(frame, copy=True)
                return None
            if self._first.shape != np.shape(frame):
                self._first = np.array(frame, copy=True)  # binning changed
                return None
            dy, dx = self.displacement(self._first, frame)
            return self._in_pattern_px(dy, dx, np.shape(frame))
        if len(frames) < 2 or np.shape(frames[0]) != np.shape(frame):
            return None
        # the previous frame came before the last move, so the measured
        # displacement includes the move: the residual corrects itself
        self._residual = self._residual + self.displacement(frames[0], frame)
        return self._in_pattern_px(*self._residual, np.shape(frame))


class AutoExposurePattern(WrappingPattern):
    """
    Keeps the bright pixels of ``channel`` at a set fraction of the camera's
    range: each timepoint the ``percentile`` of the frame is compared with
    ``target`` x ``full_scale`` and the channel's exposure is scaled toward
    it, by at most ``max_change`` (a fraction) per timepoint and within
    ``min_ms`` .. ``max_ms``. ``full_scale`` is the camera's largest count
    (default: the largest value of the image type, 65535 for 16 bits).
    """

    name = "auto_exposure"
    inner_required = False
    zoo_meta = ZooMeta(
        source="mcf10a",
        kwargs={
            "channel": "545",
            "percentile": 99,
            "target": 0.6,
            "_full_scale": 255,
            "inner": {"method": "nucleus_clamp", "channel": "545", "target": 65},
        },
        title="Auto Exposure",
        description="Scales a channel's exposure so its bright pixels sit at a set fraction of the camera's range.",
        group="adaptive",
    )

    def __init__(
        self,
        channel=None,
        percentile=99.0,
        target=0.6,
        min_ms=5.0,
        max_ms=1000.0,
        max_change=0.25,
        full_scale=None,
        **kwargs,
    ):
        """
        :param channel: the channel whose exposure is adjusted
        :param inner: the wrapped method, a ``[pattern.inner]`` table (optional)
        :param percentile: the percentile of the frame held at the target
        :param target: that percentile as a fraction of full_scale (0-1)
        :param min_ms: the shortest exposure, in ms
        :param max_ms: the longest exposure, in ms
        :param max_change: the largest change per timepoint, a fraction of the exposure
        :param full_scale: the camera's largest count (default: from the image type)
        """
        super().__init__(**kwargs)
        if channel is None:
            raise AttributeError(
                f'{self.name} must be given the channel to adjust: e.g. channel = "545"'
            )
        if not 0 < percentile <= 100:
            raise ValueError(f"{self.name}: percentile must be in (0, 100]")
        if not 0 < target < 1:
            raise ValueError(f"{self.name}: target must be between 0 and 1")
        if not 0 < min_ms <= max_ms:
            raise ValueError(f"{self.name}: need 0 < min_ms <= max_ms")
        if not 0 < max_change:
            raise ValueError(f"{self.name}: max_change must be positive")
        if full_scale is not None and full_scale <= 0:
            raise ValueError(f"{self.name}: full_scale must be positive")
        self.channel = channel
        self.percentile = float(percentile)
        self.target = float(target)
        self.min_ms = float(min_ms)
        self.max_ms = float(max_ms)
        self.max_change = float(max_change)
        self.full_scale = full_scale
        self.add_requirement(channel, raw=True)

    def next_exposure(self, frame: np.ndarray, exposure_ms: float) -> float:
        """The exposure (ms) to use next, given a frame taken at ``exposure_ms``."""
        frame = np.asarray(frame)
        full = self.full_scale
        if full is None:
            full = (
                np.iinfo(frame.dtype).max
                if np.issubdtype(frame.dtype, np.integer)
                else 1.0
            )
        level = float(np.percentile(frame, self.percentile)) / float(full)
        if level <= 0:
            wanted = exposure_ms * (1 + self.max_change)
        else:
            wanted = exposure_ms * self.target / level
        low = exposure_ms * max(0.0, 1 - self.max_change)
        high = exposure_ms * (1 + self.max_change)
        wanted = min(max(wanted, low), high)
        return float(min(max(wanted, self.min_ms), self.max_ms))

    def generate(self, context):
        pattern = self.inner_pattern(context)
        current = float(context.settings(self.channel)["exposure_ms"])
        new = self.next_exposure(context.raw(self.channel), current)
        if abs(new - current) > 0.01 * current:
            context.set_exposure(self.channel, round(new, 3))
        return pattern


class SettingsSchedulePattern(WrappingPattern):
    """
    Changes one acquisition setting of ``channel`` (``"stimulation"`` by
    default) on a schedule: ``steps`` is a list of ``[start_min, value]``
    (minutes from ``start_min``), and each value is requested when its
    minute is reached, taking effect from the next timepoint. ``setting``
    is ``"exposure"`` (value in ms), ``"config"`` (a preset of ``group``) or
    ``"property"`` (``device`` / ``property``, a laser power, say).
    """

    name = "settings_schedule"
    inner_required = False
    SETTINGS = ("exposure", "config", "property")
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "setting": "exposure",
            "steps": [[0, 50], [120, 100], [240, 200]],
            "inner": {"method": "full_on"},
        },
        time_seconds=7500,
        title="Settings Schedule",
        description="Steps an exposure, a preset or a device property through [start_min, value] pairs, around any method.",
        group="adaptive",
    )

    def __init__(
        self,
        channel="stimulation",
        setting=None,
        steps=(),
        group=None,
        device=None,
        property=None,
        **kwargs,
    ):
        """
        :param channel: the channel changed ("stimulation" or an imaging channel)
        :param inner: the wrapped method, a ``[pattern.inner]`` table (optional)
        :param setting: "exposure", "config" or "property"
        :param steps: list of [start_min, value]
        :param group: the config group ("config")
        :param device: the device ("property")
        :param property: its property ("property")
        """
        super().__init__(**kwargs)
        if setting not in self.SETTINGS:
            raise ValueError(
                f"{self.name}: setting must be one of "
                f"{', '.join(map(repr, self.SETTINGS))}, got {setting!r}"
            )
        if setting == "config" and group is None:
            raise ValueError(f'{self.name}: setting = "config" needs group')
        if setting == "property" and (device is None or property is None):
            raise ValueError(
                f'{self.name}: setting = "property" needs device and property'
            )
        parsed = []
        for step in steps:
            try:
                minute, value = step
            except (TypeError, ValueError):
                raise ValueError(
                    f"{self.name}: each step must be a [start_min, value] pair, "
                    f"got {step!r}"
                ) from None
            if setting == "exposure" and float(value) <= 0:
                raise ValueError(f"{self.name}: an exposure must be positive")
            parsed.append((float(minute), value))
        if not parsed:
            raise ValueError(
                f"{self.name}: steps must list at least one [start_min, value]"
            )
        self.steps = sorted(parsed, key=lambda s: s[0])
        self.channel = channel
        self.setting = setting
        self.group = group
        self.device = device
        self.property = property
        self._requested = None

    def value_at(self, t_minutes: float):
        """The value of the last step reached at ``t_minutes`` (None before the first)."""
        t_rel = t_minutes - self.start_min
        value = None
        for minute, v in self.steps:
            if minute <= t_rel:
                value = v
        return value

    def generate(self, context):
        pattern = self.inner_pattern(context)
        value = self.value_at(context.time / 60)
        if value is not None and value != self._requested:
            if self.setting == "exposure":
                context.set_exposure(self.channel, float(value))
            elif self.setting == "config":
                context.set_config(self.channel, self.group, str(value))
            else:
                context.set_property(self.channel, self.device, self.property, value)
            self._requested = value
        return pattern
