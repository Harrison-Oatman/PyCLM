import numpy as np

from .pattern import PatternMethod, DataDock


class CirclePattern(PatternMethod):
    """
    Projects a circle of radius rad in the center of the camera
    """

    name = "circle"

    def __init__(self, experiment_name, camera_properties, rad=1, **kwargs):
        super().__init__(experiment_name, camera_properties)

        self.rad = rad

    def initialize(self, experiment):

        super().initialize(experiment)

        return []

    def generate(self, data_dock: DataDock):

        h, w = self.pattern_shape

        center_x = self.pixel_size_um * w / 2.
        center_y = self.pixel_size_um * h / 2.

        xx, yy = self.get_meshgrid()

        print(h, w)

        # pattern = np.ones((int(h), int(w)), dtype=np.float16)

        return (((xx - center_x)**2 + (yy - center_y)**2) < (self.rad**2)).astype(np.float16)


class FullOnPattern(PatternMethod):
    """
    Turns on the entire ROI
    """

    name = "full_on"

    def __init__(self, experiment_name, camera_properties, **kwargs):
        super().__init__(experiment_name, camera_properties)

    def initialize(self, experiment):
        super().initialize(experiment)
        return []

    def generate(self, data_dock: DataDock):

        h, w = self.pattern_shape
        pattern = np.ones((int(h), int(w)), dtype=np.float16)

        return pattern
