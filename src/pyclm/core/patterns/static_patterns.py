from typing import ClassVar

import numpy as np

from .pattern import PatternMethod
from .zoo import ZooMeta


class CirclePattern(PatternMethod):
    """
    Projects a filled circle of radius ``radius_um`` in the center of the camera
    """

    name = "circle"
    renamed_kwargs: ClassVar[dict[str, str]] = {"rad": "radius_um"}
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"radius_um": 60},
        title="Circle",
        description="Filled circle centred on the camera field of view.",
        group="static",
    )

    def __init__(self, radius_um=1.0, **kwargs):
        """
        :param radius_um: radius of the circle, in um
        """
        super().__init__(**kwargs)

        if radius_um <= 0:
            raise ValueError(
                f"{self.name}: radius_um must be positive, got {radius_um}"
            )
        self.radius_um = radius_um

    def generate(self, context):
        h, w = self.pattern_shape

        center_x = self.pixel_size_um * w / 2.0
        center_y = self.pixel_size_um * h / 2.0

        xx, yy = self.get_um_meshgrid()

        return (
            ((xx - center_x) ** 2 + (yy - center_y) ** 2) < (self.radius_um**2)
        ).astype(np.float16)


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
        group="static",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, context):
        h, w = self.pattern_shape
        pattern = np.ones((int(h), int(w)), dtype=np.float16)

        return pattern
