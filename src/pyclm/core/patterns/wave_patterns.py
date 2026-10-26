from typing import ClassVar

import numpy as np

from .bar_patterns import _check_stripes, _StripeKwargs
from .pattern import PatternMethod
from .zoo import ZooMeta


class WavePatternBase(_StripeKwargs, PatternMethod):
    """
    Creates a WavePattern or StationaryWavePattern depending on the requested speed
    """

    renamed_kwargs: ClassVar[dict[str, str]] = {"wave_speed": "speed_um_per_min"}

    @classmethod
    def migrate_kwargs(cls, kwargs):
        kwargs, notes = super().migrate_kwargs(kwargs)
        if "direction" in kwargs:
            # the old direction = -1 (inward) is now a negative speed
            direction = kwargs.pop("direction")
            speed = kwargs.get("speed_um_per_min", 1.0)
            kwargs["speed_um_per_min"] = speed * (1 if direction >= 0 else -1)
            notes.append(
                "direction is gone: a negative speed_um_per_min moves the rings "
                f"inward (speed_um_per_min = {kwargs['speed_um_per_min']:g})"
            )
        return kwargs, notes

    def __new__(cls, *args, **kwargs):
        if cls is WavePatternBase:  # Check if the base class is being instantiated
            if kwargs.get("speed_um_per_min") != 0:
                return super().__new__(WavePattern)
            else:
                return super().__new__(StationaryWavePattern)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class StationaryWavePattern(WavePatternBase):
    """
    Wave coming out from the center
    """

    name = "rings (stationary)"

    def __init__(self, width_um=6.0, spacing_um=30.0, speed_um_per_min=0, **kwargs):
        """
        :param width_um: width of each ring, in um
        :param spacing_um: distance from one ring to the next, in um
        :param speed_um_per_min: 0 (moving rings are ``rings`` with a non-zero speed)
        """
        super().__init__(**kwargs)
        _check_stripes(self.name, width_um, spacing_um)
        self.width_um = width_um
        self.spacing_um = spacing_um
        self.speed_um_per_min = 0

    def generate(self, context):
        xx, yy = self.get_um_meshgrid()
        center_x, center_y = self.center_um()
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        is_on = ((distance / self.spacing_um) % 1.0) < (self.width_um / self.spacing_um)

        return is_on.astype(np.float16)


class WavePattern(WavePatternBase):
    """
    moves a wave in or out of the center
    """

    name = "rings"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"width_um": 20, "spacing_um": 100, "speed_um_per_min": 10},
        title="Rings",
        description="Concentric rings propagating outward from the center (a negative speed: inward; 0: still).",
        group="moving",
        times=(0, 150, 300, 450),
    )

    def __init__(self, width_um=6.0, spacing_um=30.0, speed_um_per_min=1.0, **kwargs):
        """
        :param width_um: width of each ring, in um
        :param spacing_um: distance from one ring to the next, in um
        :param speed_um_per_min: speed in um/min, outward; negative moves the
            rings inward, 0 holds them still
        """
        super().__init__(**kwargs)
        _check_stripes(self.name, width_um, spacing_um)
        if speed_um_per_min == 0:
            raise ValueError(
                "speed_um_per_min must be non-zero (0 selects the stationary rings)"
            )

        self.width_um = width_um
        self.spacing_um = spacing_um
        self.speed_um_per_min = speed_um_per_min

    def generate(self, context):
        t = context.time / 60

        speed = abs(self.speed_um_per_min)
        sign = 1 if self.speed_um_per_min > 0 else -1
        period_time = self.spacing_um / speed  # minutes
        duty_cycle = self.width_um / self.spacing_um

        xx, yy = self.get_um_meshgrid()
        center_x, center_y = self.center_um()
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        is_on = (
            (t * sign) - (distance / speed)
        ) % period_time < duty_cycle * period_time

        return is_on.astype(np.float16)
