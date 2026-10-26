from typing import ClassVar

import numpy as np

from .pattern import DataDock, PatternMethod
from .zoo import ZooMeta

# the defaults of the old duty_cycle / period pair (width = duty x period)
_OLD_DUTY_CYCLE = 0.2
_OLD_PERIOD_UM = 30.0


def _migrate_stripe_kwargs(cls, kwargs, notes, old_period=_OLD_PERIOD_UM):
    """
    The stripe geometry used to be ``duty_cycle`` (fraction lit) and
    ``period`` (um); it is now ``width_um`` and ``spacing_um``. Shared by
    the bar family, ``sawtooth`` and ``rings``.
    """
    if "duty_cycle" not in kwargs and "period" not in kwargs:
        return kwargs
    if "width_um" in kwargs or "spacing_um" in kwargs:
        raise ValueError(
            f"{cls.name}: give width_um and spacing_um, or the old "
            "duty_cycle and period, not both"
        )
    duty = kwargs.pop("duty_cycle", _OLD_DUTY_CYCLE)
    period = kwargs.pop("period", old_period)
    kwargs["width_um"] = duty * period
    kwargs["spacing_um"] = period
    notes.append(
        f"duty_cycle and period are now width_um = {duty * period:g} "
        f"and spacing_um = {period:g}"
    )
    return kwargs


def _check_stripes(name, width_um, spacing_um):
    if spacing_um <= 0:
        raise ValueError(f"{name}: spacing_um must be positive, got {spacing_um}")
    if not 0 <= width_um <= spacing_um:
        raise ValueError(
            f"{name}: width_um must be between 0 and spacing_um "
            f"({spacing_um}), got {width_um}"
        )


def _stripe_is_on(yy, period_um, duty_cycle, offset_um=0.0):
    """
    Where a stationary periodic bar is lit: stripes of width
    ``duty_cycle * period_um`` along the y-axis, repeating every
    ``period_um``, with the first stripe edge at ``offset_um``.
    """
    return (((yy - offset_um) / period_um) % 1.0) < duty_cycle


def _bar_is_on(yy, t_minutes, bar_speed, period_um, duty_cycle):
    """
    Where a periodic bar is lit at ``t_minutes``: the bar travels along +y at
    ``bar_speed`` um/min, or along -y when the speed is negative, repeating
    every ``period_um``. The period in time is taken from the speed's
    magnitude so the modulo is always positive.
    """
    period_time = period_um / abs(bar_speed)  # minutes
    return ((t_minutes - yy / bar_speed) % period_time) < duty_cycle * period_time


class _StripeKwargs:
    """Mixin: the old stripe and speed keywords of the bar family."""

    renamed_kwargs: ClassVar[dict[str, str]] = {
        "bar_speed": "speed_um_per_min",
    }

    @classmethod
    def migrate_kwargs(cls, kwargs):
        kwargs, notes = super().migrate_kwargs(kwargs)
        return _migrate_stripe_kwargs(cls, kwargs, notes), notes


class BarPatternBase(_StripeKwargs, PatternMethod):
    """
    Creates a BarPattern or StationaryBarPattern depending on the requested speed
    """

    def __new__(cls, *args, **kwargs):
        if cls is BarPatternBase:  # Check if the base class is being instantiated
            if kwargs.get("speed_um_per_min") != 0:
                return super().__new__(BarPattern)
            else:
                return super().__new__(StationaryBarPattern)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class StationaryBarPattern(BarPatternBase):
    """
    Bar does not move
    """

    name = "bar (stationary)"

    def __init__(self, width_um=6.0, spacing_um=30.0, speed_um_per_min=0, **kwargs):
        """
        :param width_um: width of each stripe, in um
        :param spacing_um: distance from one stripe to the next, in um
        :param speed_um_per_min: 0 (a moving bar is ``bar`` with a non-zero speed)
        """
        super().__init__(**kwargs)
        _check_stripes(self.name, width_um, spacing_um)
        self.width_um = width_um
        self.spacing_um = spacing_um
        self.speed_um_per_min = 0

    def generate(self, context):
        _xx, yy = self.get_um_meshgrid()
        is_on = _stripe_is_on(yy, self.spacing_um, self.width_um / self.spacing_um)
        return is_on.astype(np.float16)


class BarPattern(BarPatternBase):
    """
    moves a bar along the y-axis
    """

    name = "bar"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"width_um": 20, "spacing_um": 100, "speed_um_per_min": 1.0},
        title="Moving Bar",
        description="Periodic bar sweeping along the y-axis at constant speed; a negative speed reverses it and 0 holds it still.",
        group="moving",
        times=(0, 1500, 3000, 4500),
    )

    def __init__(self, width_um=6.0, spacing_um=30.0, speed_um_per_min=1.0, **kwargs):
        """
        :param width_um: width of each stripe, in um
        :param spacing_um: distance from one stripe to the next, in um; each
            point is lit for width_um / spacing_um of the time
        :param speed_um_per_min: speed in um/min, towards the bottom of the
            image (+y); negative moves the bar up, 0 holds it still
        """
        super().__init__(**kwargs)
        if speed_um_per_min == 0:
            raise ValueError(
                "speed_um_per_min must be non-zero (0 selects the stationary bar)"
            )
        _check_stripes(self.name, width_um, spacing_um)
        self.width_um = width_um
        self.spacing_um = spacing_um
        self.speed_um_per_min = speed_um_per_min  # negative moves the bar the other way

    @property
    def duty_cycle(self) -> float:
        """Fraction of the time each point is lit."""
        return self.width_um / self.spacing_um

    @property
    def period_time(self) -> float:
        """Minutes for the bar to travel one spacing (follows speed changes)."""
        return self.spacing_um / abs(self.speed_um_per_min)

    def _get_pattern_at_time(self, t_minutes):
        _xx, yy = self.get_um_meshgrid()
        is_on = _bar_is_on(
            yy, t_minutes, self.speed_um_per_min, self.spacing_um, self.duty_cycle
        )
        return is_on.astype(np.float16)

    def generate(self, context):
        return self._get_pattern_at_time(context.time / 60)


class SawToothMethod(_StripeKwargs, PatternMethod):
    name = "sawtooth"
    renamed_kwargs: ClassVar[dict[str, str]] = {"inverse": "reverse"}
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"width_um": 50, "spacing_um": 100, "speed_um_per_min": 1.0},
        title="Sawtooth",
        description="The moving bar with an intensity ramp across each bar: brightest at the leading edge, or the trailing edge with reverse = true.",
        group="moving",
        times=(0, 1500, 3000, 4500),
    )

    def __init__(
        self,
        width_um=6.0,
        spacing_um=30.0,
        speed_um_per_min=1.0,
        reverse=False,
        **kwargs,
    ):
        """
        :param width_um: width of each ramp, in um
        :param spacing_um: distance from one ramp to the next, in um
        :param speed_um_per_min: speed in um/min, towards +y; negative moves it up
        :param reverse: brightest at the trailing edge instead of the leading one
        """
        super().__init__(**kwargs)
        if speed_um_per_min == 0:
            raise ValueError("speed_um_per_min must be non-zero")
        _check_stripes(self.name, width_um, spacing_um)
        self.width_um = width_um
        self.spacing_um = spacing_um
        self.speed_um_per_min = (
            speed_um_per_min  # negative moves the ramp the other way
        )
        self.reverse = reverse

    @property
    def duty_cycle(self) -> float:
        return self.width_um / self.spacing_um

    @property
    def period_time(self) -> float:
        return self.spacing_um / abs(self.speed_um_per_min)

    def generate(self, context):
        t = context.time / 60

        _xx, yy = self.get_um_meshgrid()

        is_on = _bar_is_on(
            yy, t, self.speed_um_per_min, self.spacing_um, self.duty_cycle
        )

        val = ((t - (yy / self.speed_um_per_min)) % self.period_time) / (
            self.duty_cycle * self.period_time
        )

        if not self.reverse:
            val = 1 - val

        pattern_out = (is_on * val).astype(np.float16)

        return pattern_out


class BouncingBarPattern(BarPattern):
    name = "bar_bounce"
    renamed_kwargs: ClassVar[dict[str, str]] = {"t_loop": "loop_min"}
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "width_um": 20,
            "spacing_um": 100,
            "speed_um_per_min": 2.0,
            "loop_min": 100,
        },
        title="Bouncing Bar",
        description="The moving bar, reversing direction every half of loop_min minutes.",
        group="moving",
        times=(0, 1500, 3000, 4500),
    )

    def __init__(
        self,
        width_um=6.0,
        spacing_um=30.0,
        speed_um_per_min=1.0,
        loop_min=60.0,
        **kwargs,
    ):
        """
        :param width_um: width of each stripe, in um
        :param spacing_um: distance from one stripe to the next, in um
        :param speed_um_per_min: speed in um/min
        :param loop_min: minutes for one reversal cycle (there and back)
        """
        super().__init__(
            width_um=width_um,
            spacing_um=spacing_um,
            speed_um_per_min=speed_um_per_min,
            **kwargs,
        )
        if loop_min <= 0:
            raise ValueError(f"{self.name}: loop_min must be positive, got {loop_min}")
        self.loop_min = loop_min

    def generate(self, context):
        t = context.time % (self.loop_min * 60)

        halfway = self.loop_min * 60 / 2

        if t > halfway:
            t = halfway - (t - halfway)

        return self._get_pattern_at_time(t / 60)


class SpokesPattern(PatternMethod):
    name = "spokes"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"count": 12, "speed_deg_per_min": 1.15, "width_um": 20},
        title="Spokes",
        description="Spoke-like bars rotating around the center of the field (positive speed: counterclockwise).",
        group="moving",
        times=(0, 600, 1200, 1800),
    )

    def __init__(
        self,
        count: int = 5,
        speed_deg_per_min: float = 1.0,
        width_um: float = 20.0,
        **kwargs,
    ):
        """
        ``count`` bars of fixed width, pointing out from the center of the
        field, rotating at ``speed_deg_per_min``.

        :param count: number of spokes
        :param speed_deg_per_min: rotation speed in degrees per minute;
            positive is counterclockwise as seen on the image, negative clockwise
        :param width_um: width of each spoke, in um
        """
        super().__init__(**kwargs)
        if int(count) < 1:
            raise ValueError(f"{self.name}: count must be at least 1, got {count}")
        self.count = int(count)
        self.speed_deg_per_min = speed_deg_per_min
        self.width_um = width_um

    @classmethod
    def migrate_kwargs(cls, kwargs):
        kwargs, notes = super().migrate_kwargs(kwargs)
        if "num_bars" in kwargs:
            kwargs["count"] = kwargs.pop("num_bars")
            notes.append("num_bars is now count")
        if "bar_width" in kwargs:
            kwargs["width_um"] = kwargs.pop("bar_width")
            notes.append("bar_width is now width_um")
        if "angular_velocity" in kwargs or "angular_velocity_rad" in kwargs:
            if "speed_deg_per_min" in kwargs:
                raise ValueError(
                    f"{cls.name}: give speed_deg_per_min or the old "
                    "angular_velocity, not both"
                )
            rad = kwargs.pop("angular_velocity_rad", None)
            um = kwargs.pop("angular_velocity", 1.0)
            if not rad:
                # the old speed: um/min of a point 100 um from the center
                rad = um / 100.0
            # the old angle grew clockwise on the image; the new one counterclockwise
            deg = -float(np.degrees(rad))
            kwargs["speed_deg_per_min"] = deg
            notes.append(
                f"angular_velocity is now speed_deg_per_min = {deg:g} "
                "(degrees per minute, positive counterclockwise)"
            )
        return kwargs, notes

    def generate(self, context) -> np.ndarray:
        t = context.time / 60.0

        xx, yy = self.get_um_meshgrid()
        center_x, center_y = self.center_um()

        xx = xx - center_x
        yy = yy - center_y

        output = xx * 0.0

        # image rows grow downwards, so a counterclockwise turn on the image
        # decreases the angle measured in (x, y) image coordinates
        rotation = -np.radians(self.speed_deg_per_min) * t
        for spoke_n in range(self.count):
            angles_between = 2 * np.pi / self.count
            spoke_theta = spoke_n * angles_between + rotation

            spoke_x = np.cos(spoke_theta)
            spoke_y = np.sin(spoke_theta)

            projection_mag = xx * spoke_x + yy * spoke_y
            projection_x = projection_mag * spoke_x
            projection_y = projection_mag * spoke_y

            projection_distance = np.sqrt(
                (xx - projection_x) ** 2 + (yy - projection_y) ** 2
            )

            output += (projection_mag > 0) & (
                np.abs(projection_distance) < (self.width_um / 2)
            )

        return output > 0
