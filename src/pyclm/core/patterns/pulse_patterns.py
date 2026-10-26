"""
Open-loop pulse trains: stimulation that varies in time but never moves.

``global_pulse`` gates the whole field on a period; ``pulse_schedule`` gates
it on an explicit list of intervals; ``blinking_bar`` gates the stationary
bar's stripes with the same temporal gate as ``global_pulse``; ``pulsed``
gates any other method with it; ``chirp`` pulses the whole field with a
period that changes over the run.
"""

from typing import ClassVar

import numpy as np

from .bar_patterns import _check_stripes, _stripe_is_on
from .composite import WrappingPattern
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
    train the temporal twin of a ``bar``, set ``reference_speed_um_per_min`` to that
    bar's speed instead and the phase is derived from the field geometry: see
    :attr:`effective_phase_min`.
    """

    name = "global_pulse"
    renamed_kwargs: ClassVar[dict[str, str]] = {
        "reference_bar_speed": "reference_speed_um_per_min"
    }
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"period_min": 100, "on_min": 20, "start_min": 60},
        time_seconds=3900,
        title="Global Pulse",
        description="The whole field on for on_min out of every period_min, between start_min and stop_min.",
        group="timed",
        timeline=(0, 30000),
    )

    def __init__(
        self,
        period_min=100.0,
        on_min=20.0,
        phase_min=0.0,
        reference_speed_um_per_min=None,
        **kwargs,
    ):
        """
        :param period_min: minutes between successive on-edges
        :param on_min: minutes the field stays on at each on-edge
        :param phase_min: minutes from ``start_min`` to the first on-edge
        :param reference_speed_um_per_min: speed of the ``bar`` this train twins;
                                    phases the first on-edge to that bar
                                    reaching the field centre
        :param start_min: experiment minute stimulation begins
        :param stop_min: experiment minute stimulation ends (None: never)
        """
        super().__init__(**kwargs)

        _check_gate(period_min, on_min, self.name)
        if reference_speed_um_per_min == 0:
            raise ValueError(
                f"{self.name}: reference_speed_um_per_min must be non-zero "
                "(a stationary bar has no arrival time)"
            )

        self.period_min = period_min
        self.on_min = on_min
        self.phase_min = phase_min
        self.reference_speed_um_per_min = reference_speed_um_per_min

    @property
    def effective_phase_min(self) -> float:
        """
        The phase actually used, in minutes after ``start_min``.

        With ``reference_speed_um_per_min`` set, ``phase_min`` is added to the minute
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
        if self.reference_speed_um_per_min:
            _center_x, center_y = self.center_um()
            phase += center_y / self.reference_speed_um_per_min
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
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"pulses": [[60, 20], [180, 5], [240, 60]]},
        time_seconds=3900,
        title="Pulse Schedule",
        description="The whole field on during an explicit list of [start_min, duration_min] pulses.",
        group="timed",
        timeline=(0, 21600),
    )

    def __init__(self, pulses=(), **kwargs):
        """
        :param pulses: list of ``[start_min, duration_min]`` pairs, each
                       measured from ``start_min``
        :param start_min: experiment minute the schedule is measured from
            (and before which the field is dark)
        """
        super().__init__(**kwargs)

        self.pulses = [self._check_pulse(p) for p in pulses]

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
    (stripes of width ``width_um``, repeating every
    ``spacing_um``, the first edge at ``offset_um``) ANDed with the temporal
    gate of ``global_pulse``. ``on_min = period_min`` is the limiting case of
    stripes that never blink.
    """

    name = "blinking_bar"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"width_um": 20, "spacing_um": 100, "period_min": 100, "on_min": 20},
        title="Blinking Bar",
        description="Stationary stripes gated on and off by a periodic pulse train.",
        group="timed",
        timeline=(0, 30000),
    )

    @classmethod
    def migrate_kwargs(cls, kwargs):
        kwargs, notes = super().migrate_kwargs(kwargs)
        if "duty_cycle" in kwargs:
            if "width_um" in kwargs:
                raise ValueError(
                    f"{cls.name}: give width_um or the old duty_cycle, not both"
                )
            width = kwargs.pop("duty_cycle") * kwargs.get("spacing_um", 100.0)
            kwargs["width_um"] = width
            notes.append(f"duty_cycle is now width_um = {width:g}")
        return kwargs, notes

    def __init__(
        self,
        width_um=20.0,
        spacing_um=100.0,
        period_min=100.0,
        on_min=20.0,
        offset_um=0.0,
        phase_min=0.0,
        **kwargs,
    ):
        """
        :param width_um: width of each stripe, in um
        :param spacing_um: stripe period along the bar axis, in um
        :param period_min: minutes between successive on-edges
        :param on_min: minutes the stripes stay on at each on-edge
        :param offset_um: position of the first stripe edge, in um
        :param phase_min: minutes from ``start_min`` to the first on-edge
        :param start_min: experiment minute stimulation begins
        :param stop_min: experiment minute stimulation ends (None: never)
        """
        super().__init__(**kwargs)

        _check_gate(period_min, on_min, self.name)
        _check_stripes(self.name, width_um, spacing_um)

        self.width_um = width_um
        self.spacing_um = spacing_um
        self.period_min = period_min
        self.on_min = on_min
        self.offset_um = offset_um
        self.phase_min = phase_min

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

        return _stripe_is_on(
            yy, self.spacing_um, self.width_um / self.spacing_um, self.offset_um
        )

    def generate(self, context):
        if not self.is_on(context.time / 60):
            h, w = self.pattern_shape
            return np.zeros((int(h), int(w)), dtype=np.float16)

        return self.spatial_mask().astype(np.float16)


class PulsedPattern(WrappingPattern):
    """
    Any method gated by a pulse train: the inner method's pattern for
    ``on_min`` out of every ``period_min``, dark in between, the first
    on-edge ``phase_min`` after ``start_min``. The inner method runs at
    every timepoint either way, so a closed loop inside keeps up with the
    cells while the light is off. ``global_pulse`` is ``pulsed`` around
    ``full_on``.
    """

    name = "pulsed"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "period_min": 60,
            "on_min": 20,
            "inner": {"method": "circle", "radius_um": 120},
        },
        title="Pulsed",
        description="Gates any method (the [pattern.inner] table) on for on_min out of every period_min.",
        group="timed",
        timeline=(0, 18000),
    )

    def __init__(self, period_min=60.0, on_min=20.0, phase_min=0.0, **kwargs):
        """
        :param period_min: minutes between successive on-edges
        :param on_min: minutes the inner pattern is shown at each on-edge
        :param phase_min: minutes from ``start_min`` to the first on-edge
        """
        super().__init__(**kwargs)
        _check_gate(period_min, on_min, self.name)
        self.period_min = float(period_min)
        self.on_min = float(on_min)
        self.phase_min = float(phase_min)

    def is_on(self, t_minutes) -> bool:
        return _gate_is_on(
            t_minutes,
            self.period_min,
            self.on_min,
            self.phase_min,
            self.start_min,
            self.stop_min,
        )

    def generate(self, context):
        pattern = self.inner_pattern(context)
        if not self.is_on(context.time / 60):
            return np.zeros_like(pattern)
        return pattern


class ChirpPattern(PatternMethod):
    """
    The whole field pulsed with a period that sweeps from
    ``period_start_min`` to ``period_end_min`` over ``sweep_min`` (measured
    from ``start_min``) and then stays at ``period_end_min``. Each period is
    on for its first ``duty_cycle`` fraction. ``sweep = "log"`` changes the
    period by the same factor every minute (equal time per octave);
    ``"linear"`` by the same number of minutes every minute.
    """

    name = "chirp"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "period_start_min": 60,
            "period_end_min": 10,
            "duty_cycle": 0.5,
            "sweep_min": 300,
        },
        title="Chirp",
        description="Whole-field pulses whose period sweeps from period_start_min to period_end_min: a frequency response in one run.",
        group="timed",
        timeline=(0, 21600),
    )

    SWEEPS = ("log", "linear")

    def __init__(
        self,
        period_start_min=None,
        period_end_min=None,
        duty_cycle=0.5,
        sweep_min=None,
        sweep="log",
        **kwargs,
    ):
        """
        :param period_start_min: the period at the start of the sweep, in minutes
        :param period_end_min: the period at its end (and after), in minutes
        :param duty_cycle: fraction of each period the field is on (0-1)
        :param sweep_min: duration of the sweep, in minutes
        :param sweep: "log" or "linear"
        """
        super().__init__(**kwargs)
        for key, value in (
            ("period_start_min", period_start_min),
            ("period_end_min", period_end_min),
            ("sweep_min", sweep_min),
        ):
            if value is None or value <= 0:
                raise ValueError(f"{self.name}: {key} must be positive, got {value}")
        if not 0.0 <= duty_cycle <= 1.0:
            raise ValueError(
                f"{self.name}: duty_cycle must be in 0-1, got {duty_cycle}"
            )
        if sweep not in self.SWEEPS:
            raise ValueError(
                f"{self.name}: sweep must be 'log' or 'linear', got {sweep!r}"
            )
        self.period_start_min = float(period_start_min)
        self.period_end_min = float(period_end_min)
        self.duty_cycle = float(duty_cycle)
        self.sweep_min = float(sweep_min)
        self.sweep = sweep

    def period_at(self, t_rel_min: float) -> float:
        """The instantaneous period ``t_rel_min`` after the sweep starts."""
        p0, p1, s = self.period_start_min, self.period_end_min, self.sweep_min
        f = min(max(t_rel_min, 0.0), s) / s
        if self.sweep == "log":
            return p0 * (p1 / p0) ** f
        return p0 + (p1 - p0) * f

    def cycles(self, t_rel_min: float) -> float:
        """Periods completed ``t_rel_min`` after the start: the integral of 1 / period."""
        p0, p1, s = self.period_start_min, self.period_end_min, self.sweep_min
        t = max(t_rel_min, 0.0)
        in_sweep = min(t, s)
        if p0 == p1:
            n = in_sweep / p0
        elif self.sweep == "log":
            r = p1 / p0
            n = s / (p0 * np.log(r)) * (1.0 - r ** (-in_sweep / s))
        else:
            k = (p1 - p0) / s
            n = np.log((p0 + k * in_sweep) / p0) / k
        return float(n + max(t - s, 0.0) / p1)

    def is_on(self, t_minutes) -> bool:
        if t_minutes < self.start_min:
            return False
        return (self.cycles(t_minutes - self.start_min) % 1.0) < self.duty_cycle

    def generate(self, context):
        h, w = self.pattern_shape
        return np.full(
            (int(h), int(w)), self.is_on(context.time / 60), dtype=np.float16
        )
