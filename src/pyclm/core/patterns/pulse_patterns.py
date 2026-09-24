"""
Open-loop pulse trains: stimulation that varies in time but never moves.

``global_pulse`` gates the whole field on a period; ``pulse_schedule`` gates
it on an explicit list of intervals; ``blinking_bar`` gates the stationary
bar's stripes with the same temporal gate as ``global_pulse``.
"""

import numpy as np

from .bar_patterns import _stripe_is_on
from .pattern import PatternMethod
from .zoo import ZooMeta


def _in_window(t_minutes, start_min, stop_min) -> bool:
    """Whether ``t_minutes`` lies in ``[start_min, stop_min)``; no ``stop_min`` means no end."""
    if t_minutes < start_min:
        return False
    return stop_min is None or t_minutes < stop_min


def _gate_is_on(t_minutes, period_min, on_min, phase_min, start_min, stop_min) -> bool:
    """
    Whether a periodic gate is open at ``t_minutes``: on for ``on_min`` out of
    every ``period_min``, the first on-edge at ``start_min + phase_min``, and
    nothing at all outside ``[start_min, stop_min)``.
    """
    if not _in_window(t_minutes, start_min, stop_min):
        return False
    t_rel = (t_minutes - start_min - phase_min) % period_min
    return t_rel < on_min


def _check_gate(period_min, on_min, name):
    if period_min <= 0:
        raise ValueError(f"{name}: period_min must be positive, got {period_min}")
    if not 0 <= on_min <= period_min:
        raise ValueError(
            f"{name}: on_min must be between 0 and period_min "
            f"({period_min}), got {on_min}"
        )


class GlobalPulsePattern(PatternMethod):
    """
    Spatially uniform pulse train: the whole field on for ``on_min`` out of
    every ``period_min``, within ``[start_min, stop_min)``.

    ``phase_min`` offsets the first on-edge from ``start_min``. To make the
    train the temporal twin of a ``bar``, set ``reference_bar_speed`` to that
    bar's speed instead and the phase is derived from the field geometry: see
    :attr:`effective_phase_min`.
    """

    name = "global_pulse"

    def __init__(
        self,
        period_min=100.0,
        on_min=20.0,
        phase_min=0.0,
        reference_bar_speed=None,
        start_min=0.0,
        stop_min=None,
        **kwargs,
    ):
        """
        :param period_min: minutes between successive on-edges
        :param on_min: minutes the field stays on at each on-edge
        :param phase_min: minutes from ``start_min`` to the first on-edge
        :param reference_bar_speed: um/min of the ``bar`` this train twins;
                                    phases the first on-edge to that bar
                                    reaching the field centre
        :param start_min: experiment minute stimulation begins
        :param stop_min: experiment minute stimulation ends (None: never)
        """
        super().__init__(**kwargs)

        _check_gate(period_min, on_min, self.name)
        if reference_bar_speed == 0:
            raise ValueError(
                f"{self.name}: reference_bar_speed must be non-zero "
                "(a stationary bar has no arrival time)"
            )

        self.period_min = period_min
        self.on_min = on_min
        self.phase_min = phase_min
        self.reference_bar_speed = reference_bar_speed
        self.start_min = start_min
        self.stop_min = stop_min

    @property
    def effective_phase_min(self) -> float:
        """
        The phase actually used, in minutes after ``start_min``.

        With ``reference_bar_speed`` set, ``phase_min`` is added to the minute
        at which a ``bar`` of that speed reaches the centre of the field.
        ``bar`` lights a point ``y`` when ``(t - y / bar_speed) % period_time``
        is within its duty cycle, so that on-edge falls at ``y / bar_speed``;
        taking ``y`` from :meth:`~pyclm.core.patterns.pattern.PatternMethod.center_um`
        keeps the two in one micron frame, and the result needs nothing beyond
        the camera geometry PyCLM already injects. It follows a binning change,
        since the field's size in microns does not.

        This is only the twin of that bar if ``period_min`` is the bar's
        temporal period, i.e. its spacing divided by its speed.
        """
        phase = self.phase_min
        if self.reference_bar_speed:
            _centre_x, centre_y = self.center_um()
            phase += centre_y / self.reference_bar_speed
        return phase % self.period_min

    def is_on(self, t_minutes) -> bool:
        """Whether the field is lit at ``t_minutes`` of the experiment."""
        return _gate_is_on(
            t_minutes,
            self.period_min,
            self.on_min,
            self.effective_phase_min,
            self.start_min,
            self.stop_min,
        )

    def generate(self, context):
        h, w = self.pattern_shape

        return np.full(
            (int(h), int(w)), self.is_on(context.time / 60), dtype=np.float16
        )


class PulseSchedulePattern(PatternMethod):
    """
    Spatially uniform stimulation on an explicit schedule: the whole field is
    on whenever the experiment time falls inside one of the listed intervals.
    ``global_pulse`` with a list instead of a period, for isolated pulses::

        pulses = [[60, 20], [540, 20]]   # 20 min at 1 h and at 9 h
    """

    name = "pulse_schedule"

    def __init__(self, pulses=(), start_min=0.0, **kwargs):
        """
        :param pulses: list of ``[start_min, duration_min]`` pairs, each
                       measured from ``start_min``
        :param start_min: experiment minute the schedule is measured from
        """
        super().__init__(**kwargs)

        self.pulses = [self._check_pulse(p) for p in pulses]
        self.start_min = start_min

    def _check_pulse(self, pulse):
        try:
            start, duration = pulse
        except (TypeError, ValueError):
            raise ValueError(
                f"{self.name}: each pulse must be a [start_min, duration_min] "
                f"pair, got {pulse!r}"
            ) from None
        if duration <= 0:
            raise ValueError(
                f"{self.name}: pulse duration must be positive, got {duration}"
            )
        return [float(start), float(duration)]

    def is_on(self, t_minutes) -> bool:
        """Whether the field is lit at ``t_minutes`` of the experiment."""
        t_rel = t_minutes - self.start_min
        return any(start <= t_rel < start + duration for start, duration in self.pulses)

    def generate(self, context):
        h, w = self.pattern_shape

        return np.full(
            (int(h), int(w)), self.is_on(context.time / 60), dtype=np.float16
        )


class BlinkingBarPattern(PatternMethod):
    """
    Stationary stripes that blink: the spatial mask of ``bar`` at speed 0
    (stripes of width ``duty_cycle * spacing_um``, repeating every
    ``spacing_um``, the first edge at ``offset_um``) ANDed with the temporal
    gate of ``global_pulse``. ``on_min = period_min`` is the limiting case of
    stripes that never blink.
    """

    name = "blinking_bar"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"spacing_um": 100, "duty_cycle": 0.2, "period_min": 100, "on_min": 20},
        title="Blinking Bar",
        description="Stationary stripes gated on and off by a periodic pulse train.",
    )

    def __init__(
        self,
        spacing_um=100.0,
        duty_cycle=0.2,
        period_min=100.0,
        on_min=20.0,
        offset_um=0.0,
        phase_min=0.0,
        start_min=0.0,
        stop_min=None,
        **kwargs,
    ):
        """
        :param spacing_um: stripe period along the bar axis, in um
        :param duty_cycle: fraction of ``spacing_um`` that is lit (float 0-1)
        :param period_min: minutes between successive on-edges
        :param on_min: minutes the stripes stay on at each on-edge
        :param offset_um: position of the first stripe edge, in um
        :param phase_min: minutes from ``start_min`` to the first on-edge
        :param start_min: experiment minute stimulation begins
        :param stop_min: experiment minute stimulation ends (None: never)
        """
        super().__init__(**kwargs)

        _check_gate(period_min, on_min, self.name)
        if spacing_um <= 0:
            raise ValueError(f"{self.name}: spacing_um must be positive")

        self.spacing_um = spacing_um
        self.duty_cycle = duty_cycle
        self.period_min = period_min
        self.on_min = on_min
        self.offset_um = offset_um
        self.phase_min = phase_min
        self.start_min = start_min
        self.stop_min = stop_min

    def is_on(self, t_minutes) -> bool:
        """Whether the stripes are lit at ``t_minutes`` of the experiment."""
        return _gate_is_on(
            t_minutes,
            self.period_min,
            self.on_min,
            self.phase_min,
            self.start_min,
            self.stop_min,
        )

    def spatial_mask(self) -> np.ndarray:
        """The stripes, as ``bar`` draws them at speed 0."""
        _xx, yy = self.get_um_meshgrid()

        return _stripe_is_on(yy, self.spacing_um, self.duty_cycle, self.offset_um)

    def generate(self, context):
        if not self.is_on(context.time / 60):
            h, w = self.pattern_shape
            return np.zeros((int(h), int(w)), dtype=np.float16)

        return self.spatial_mask().astype(np.float16)
