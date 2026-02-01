import numpy as np

from .pattern import PatternMethod


class CirclePattern(PatternMethod):
    """
    Projects a circle of radius rad in the center of the camera
    """

    name = "circle"

    def __init__(self, rad=1, **kwargs):
        super().__init__(**kwargs)

        self.rad = rad

    def generate(self, context):
        h, w = self.pattern_shape

        center_x = self.pixel_size_um * w / 2.0
        center_y = self.pixel_size_um * h / 2.0

        xx, yy = self.get_um_meshgrid()

        print(h, w)

        return (((xx - center_x) ** 2 + (yy - center_y) ** 2) < (self.rad**2)).astype(
            np.float16
        )


class FullOnPattern(PatternMethod):
    """
    Turns on the entire ROI
    """

    name = "full_on"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, context):
        h, w = self.pattern_shape
        pattern = np.ones((int(h), int(w)), dtype=np.float16)

        return pattern
