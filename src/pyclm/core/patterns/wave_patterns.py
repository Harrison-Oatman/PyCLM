import numpy as np

from .pattern import PatternMethod


class WavePatternBase(PatternMethod):
    """
    Creates a WavePattern or StationaryWavePattern depending on the requested wavespeed
    """

    def __new__(cls, *args, **kwargs):
        if cls is WavePatternBase:  # Check if the base class is being instantiated
            if kwargs.get("wave_speed") != 0:
                return super().__new__(WavePattern)
            else:
                return super().__new__(StationaryWavePattern)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"Initializing {self.__class__.__name__}")


class StationaryWavePattern(WavePatternBase):
    """
    Wave coming out from the center
    """

    name = "wave (stationary)"

    def __init__(self, duty_cycle=0.2, wave_speed=0, period=30, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           vertical axis containing "on" pixels
        :param wave_speed: speed in um/min
        :param period: period in um
        """
        super().__init__(**kwargs)

        self.duty_cycle = duty_cycle
        self.wave_speed = 0
        self.period_space = period  # in um
        self.period_time = 0  # in minutes

    def generate(self, context):
        xx, yy = self.get_meshgrid()
        center_y, center_x = self.pattern_shape[0] / 2, self.pattern_shape[1] / 2
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        is_on = ((distance / self.period_space) % 1.0) < self.duty_cycle

        return is_on.astype(np.float16)


class WavePattern(WavePatternBase):
    """
    moves a wave in or out of the center
    """

    name = "wave"

    def __init__(self, duty_cycle=0.2, wave_speed=1, period=30, direction=1, **kwargs):
        """
        :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
                           radial axis containing "on" pixels
        :param wave_speed: speed in um/min
        :param period: period in um
        :param direction: movement in/out relative to the center. 1 is out; -1 is in
        """
        super().__init__(**kwargs)

        self.duty_cycle = duty_cycle
        self.wave_speed = wave_speed
        self.period_space = period  # in um
        self.period_time = period / wave_speed  # in minutes
        self.direction = direction

    def generate(self, context):
        t = context.time / 60

        xx, yy = self.get_meshgrid()
        center_y, center_x = self.pattern_shape[0] / 2, self.pattern_shape[1] / 2
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        is_on = (
            ((t * self.direction) - (distance / self.wave_speed)) % self.period_time
        ) < self.duty_cycle * self.period_time

        return is_on.astype(np.float16)


# class SawToothMethod(PatternMethod):

#     name = "sawtooth"

#     def __init__(self, experiment_name, camera_properties, duty_cycle=0.2,
#                  wave_speed=1, period=30, inverse=False, **kwargs):
#         """
#         :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
#                            vertical axis containing "on" pixels
#         :param wave_speed: speed in um/min
#         :param period: period in um
#         """
#         super().__init__(experiment_name, camera_properties)

#         self.duty_cycle = duty_cycle
#         self.wave_speed = wave_speed
#         self.period_space = period  # in um
#         self.period_time = period / wave_speed  # in minutes
#         self.inverse = inverse

#     def initialize(self, experiment):
#         super().initialize(experiment)

#         return []

#     def generate(self, data_dock: DataDock):
#         t = data_dock.time_seconds / 60

#         xx, yy = self.get_meshgrid()

#         is_on = ((t - (yy / self.wave_speed)) % self.period_time) < self.duty_cycle * self.period_time

#         val = ((t - (yy / self.wave_speed)) % self.period_time) / (self.duty_cycle * self.period_time)

#         if not self.inverse:
#             val = 1 - val

#         pattern_out = (is_on*val).astype(np.float16)

#         print(np.min(pattern_out), np.max(pattern_out))

#         return pattern_out


# class OscillatingWavePattern(WavePattern):

#     name = "wave_bounce"

#     def __init__(self, experiment_name, camera_properties, duty_cycle=0.2,
#                  wave_speed=1, period=30, t_loop=60, **kwargs):
#         """

#         :param duty_cycle: fraction of time spent on (float 0-1), and consequently fraction of
#                            vertical axis containing "on" pixels
#         :param wave_speed: speed in um/min
#         :param period: period in um
#         :param t_loop: period of reversal (there and back) in minutes
#         """
#         super().__init__(experiment_name, camera_properties, duty_cycle, wave_speed, period, **kwargs)
#         self.t_loop_s = t_loop * 60

#     def generate(self, data_dock: DataDock):
#         t = data_dock.time_seconds
#         t = t % self.t_loop_s

#         halfway = self.t_loop_s / 2

#         if t > halfway:
#             t = halfway - (t - halfway)

#         data_dock.time_seconds = t

#         return super().generate(data_dock)
