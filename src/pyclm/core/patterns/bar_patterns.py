import numpy as np

from .pattern import DataDock, PatternMethod


class BarPatternBase(PatternMethod):
    """
    Creates a BarPattern or StationaryBarPattern depending on the requested barspeed
    """

    def __new__(cls, *args, **kwargs):
        if cls is BarPatternBase:  # Check if the base class is being instantiated
            if kwargs.get("bar_speed") != 0:
                return super().__new__(BarPattern)
            else:
                return super().__new__(StationaryBarPattern)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"Initializing {self.__class__.__name__}")


class StationaryBarPattern(BarPatternBase):
    """
    Bar does not move
    """

    name = "bar (stationary)"

    def __init__(self, duty_cycle=0.2, bar_speed=0, period=30, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        """
        super().__init__(**kwargs)

        self.duty_cycle = duty_cycle
        self.bar_speed = 0
        self.period_space = period  # in um
        self.period_time = 0  # in minutes

    def generate(self, context):
        _xx, yy = self.get_meshgrid()

        is_on = ((yy / self.period_space) % 1.0) < self.duty_cycle

        return is_on.astype(np.float16)


class BarPattern(BarPatternBase):
    """
    moves a bar along the y-axis
    """

    name = "bar"

    def __init__(self, duty_cycle=0.2, bar_speed=1, period=30, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        """
        super().__init__(**kwargs)

        self.duty_cycle = duty_cycle
        self.bar_speed = bar_speed
        self.period_space = period  # in um
        self.period_time = period / bar_speed  # in minutes

    def _get_pattern_at_time(self, t_minutes):
        _xx, yy = self.get_meshgrid()
        is_on = (
            (t_minutes - (yy / self.bar_speed)) % self.period_time
        ) < self.duty_cycle * self.period_time
        return is_on.astype(np.float16)

    def generate(self, context):
        return self._get_pattern_at_time(context.time / 60)


class SawToothMethod(PatternMethod):
    name = "sawtooth"

    def __init__(self, duty_cycle=0.2, bar_speed=1, period=30, inverse=False, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        """
        super().__init__(**kwargs)

        self.duty_cycle = duty_cycle
        self.bar_speed = bar_speed
        self.period_space = period  # in um
        self.period_time = period / bar_speed  # in minutes
        self.inverse = inverse

    def generate(self, context):
        t = context.time / 60

        _xx, yy = self.get_meshgrid()

        is_on = (
            (t - (yy / self.bar_speed)) % self.period_time
        ) < self.duty_cycle * self.period_time

        val = ((t - (yy / self.bar_speed)) % self.period_time) / (
            self.duty_cycle * self.period_time
        )

        if not self.inverse:
            val = 1 - val

        pattern_out = (is_on * val).astype(np.float16)

        print(np.min(pattern_out), np.max(pattern_out))

        return pattern_out


class BouncingBarPattern(BarPattern):
    name = "bar_bounce"

    def __init__(self, duty_cycle=0.2, bar_speed=1, period=30, t_loop=60, **kwargs):
        """

        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        :param t_loop: period of reversal (there and back) in minutes
        """
        super().__init__(
            duty_cycle=duty_cycle, bar_speed=bar_speed, period=period, **kwargs
        )
        self.t_loop_s = t_loop * 60

    def generate(self, context):
        t = context.time
        t = t % self.t_loop_s

        halfway = self.t_loop_s / 2

        if t > halfway:
            t = halfway - (t - halfway)

        return self._get_pattern_at_time(t / 60)


class RotatingBarPattern(PatternMethod):
    name = "rotate_bar"

    def __init__(
        self,
        num_bars: int = 5,
        angular_velocity: float = 1.0,
        bar_width: float = 20,
        angular_velocity_rad: float | None = None,
        **kwargs,
    ):
        """
        Directs a specified number of bars with fixed width to rotate around the center of the
        camera view at a specified speed. Note that the default speed is in units of um/min. This
        is based on the angular velocity of a point on the bar 100um out from the center. To use rad/min,
        specify angular_velocity_rad

        :param num_bars: number of bars sticking outwards from the center
        :param angular_velocity: um/min speed of a point on the bar 100um from the center. Can be overwritten
            by angular_velocity_rad
        :param bar_width: um width of the bar
        """

        super().__init__(**kwargs)

        self.num_bars = num_bars
        self.angular_velocity = angular_velocity
        self.bar_width = bar_width
        self.angular_velocity_rad = angular_velocity_rad

        if not self.angular_velocity_rad:
            self.angular_velocity_rad = self.angular_velocity / 100.0

    def generate(self, context) -> np.ndarray:
        t = context.time / 60.0

        xx, yy = self.get_meshgrid()
        center_x, center_y = self.center_um()

        xx = xx - center_x
        yy = yy - center_y

        output = xx * 0.0

        for spoke_n in range(self.num_bars):
            angles_between = 2 * np.pi / self.num_bars
            spoke_theta = spoke_n * angles_between + self.angular_velocity_rad * t

            spoke_x = np.cos(spoke_theta)
            spoke_y = np.sin(spoke_theta)

            projection_mag = xx * spoke_x + yy * spoke_y
            projection_x = projection_mag * spoke_x
            projection_y = projection_mag * spoke_y

            projection_distance = np.sqrt(
                (xx - projection_x) ** 2 + (yy - projection_y) ** 2
            )

            output += (projection_mag > 0) & (
                np.abs(projection_distance) < (self.bar_width / 2)
            )

        return output > 0
