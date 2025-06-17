import numpy as np

from src.patterns import DataDock, PatternModel


class BarPatternBase(PatternModel):
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

    def __init__(self, experiment_name, camera_properties, duty_cycle=0.2, bar_speed=0, period=30, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        """
        super().__init__(experiment_name, camera_properties)

        self.duty_cycle = duty_cycle
        self.bar_speed = 0
        self.period_space = period    # in um
        self.period_time = 0    # in minutes

    def initialize(self, experiment):
        super().initialize(experiment)

        return []

    def generate(self, data_dock: DataDock):

        xx, yy = self.get_meshgrid()

        is_on = ((yy / self.period_space) % 1.0) < self.duty_cycle

        return is_on.astype(np.float16)


class BarPattern(BarPatternBase):
    """
    moves a bar along the y-axis
    """

    name = "bar"

    def __init__(self, experiment_name, camera_properties, duty_cycle=0.2, bar_speed=1, period=30, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        """
        super().__init__(experiment_name, camera_properties)

        self.duty_cycle = duty_cycle
        self.bar_speed = bar_speed
        self.period_space = period    # in um
        self.period_time = period / bar_speed    # in minutes

    def initialize(self, experiment):
        super().initialize(experiment)

        return []

    def generate(self, data_dock: DataDock):

        t = data_dock.t / 60

        xx, yy = self.get_meshgrid()

        is_on = ((t - (yy / self.bar_speed)) % self.period_time) < self.duty_cycle*self.period_time

        return is_on.astype(np.float16)


class BouncingBarPattern(BarPattern):

    name = "bar_bounce"

    def __init__(self, experiment_name, camera_properties, duty_cycle=0.2,
                 bar_speed=1, period=30, t_loop=60, **kwargs):
        """

        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param bar_speed: speed in um/min
        :param period: period in um
        :param t_loop: period of reversal (there and back) in minutes
        """
        super().__init__(experiment_name, camera_properties, duty_cycle, bar_speed, period, **kwargs)
        self.t_loop_s = t_loop * 60

    def generate(self, data_dock: DataDock):
        t = data_dock.t
        t = t % self.t_loop_s

        halfway = self.t_loop_s / 2

        if t > halfway:
            t = halfway - (t - halfway)

        data_dock.t = t

        return super().generate(data_dock)
