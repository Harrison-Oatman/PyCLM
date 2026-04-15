import numpy as np
from skimage.measure import label, regionprops

from .pattern import PatternContext, PatternMethod
from .zoo import ZooMeta


class PatternAlongAxis(PatternMethod):
    def __init__(self, channel="638", **kwargs):
        super().__init__(**kwargs)

        self.channel = channel
        self.add_requirement(channel, raw=False, seg=True)

    def apply_magnitude(self, mag) -> np.ndarray:
        raise NotImplementedError

    def generate(self, context: PatternContext):
        mask = context.segmentation(self.channel)

        labeled_mask = label(mask)
        props = regionprops(labeled_mask)

        if len(props) == 0:
            np.zeros_like(mask)

        biggest_prop_area = 0
        for prop in props:
            if prop.area > biggest_prop_area:
                mask = labeled_mask == prop.label
                centroid = prop.centroid
                long_axis = (np.sin(prop.orientation), np.cos(prop.orientation))
                axis_length = prop.axis_major_length

                biggest_prop_area = prop.area

        y_arange = np.arange(self.pattern_shape[0])
        x_arange = np.arange(self.pattern_shape[1])

        yy, xx = np.meshgrid(y_arange, x_arange)

        mag = (yy - centroid[1]) * long_axis[0] + (xx - centroid[0]) * long_axis[1]
        mag = np.abs(mag) / (axis_length / 2)

        included = self.apply_magnitude(mag)

        return included * mask


class InnerPatternMethod(PatternAlongAxis):
    zoo_meta = ZooMeta(
        source="fly",
        title="Inner fraction",
        description="Applies a bar to the center of the long axis",
        kwargs={"fraction_length": 0.1},
    )

    name = "ap_inner"

    def __init__(self, fraction_length=0.1, **kwargs):
        super().__init__(**kwargs)

        self.fraction_length = fraction_length

    def apply_magnitude(self, mag) -> np.ndarray:
        return mag < self.fraction_length


class OuterPatternMethod(PatternAlongAxis):
    zoo_meta = ZooMeta(
        source="fly",
        title="Outer fraction",
        description="Applies a bar to the poles of the long axis",
        kwargs={"fraction_length": 0.1},
    )

    name = "ap_outer"

    def __init__(self, fraction_length=0.1, **kwargs):
        super().__init__(**kwargs)

        self.fraction_length = fraction_length

    def apply_magnitude(self, mag) -> np.ndarray:
        return mag > (1 - self.fraction_length)
