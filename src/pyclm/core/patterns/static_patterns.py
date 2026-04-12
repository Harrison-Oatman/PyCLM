import numpy as np

from .pattern import PatternMethod
from .zoo import ZooMeta


class CirclePattern(PatternMethod):
    """
    Projects a circle of radius rad in the center of the camera
    """

    name = "circle"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"rad": 60},
        title="Circle",
        description="Filled circle centred on the camera field of view.",
    )

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
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={},
        title="Full On",
        description="Illuminates the entire camera field of view.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, context):
        h, w = self.pattern_shape
        pattern = np.ones((int(h), int(w)), dtype=np.float16)

        return pattern
