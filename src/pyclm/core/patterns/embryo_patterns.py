import logging
from typing import ClassVar

import numpy as np
from skimage.measure import label, regionprops

from .pattern import PatternContext, PatternMethod
from .zoo import ZooMeta

logger = logging.getLogger(__name__)


class PatternAlongAxis(PatternMethod):
    # channel used to default to "638"; old TOMLs that leave it out still run
    legacy_defaults: ClassVar[dict[str, object]] = {"channel": "638"}
    renamed_kwargs: ClassVar[dict[str, str]] = {"fraction_length": "fraction"}

    def __init__(self, channel=None, **kwargs):
        """
        :param channel: the channel whose segmentation holds the embryo (the
            largest object is used)
        """
        super().__init__(**kwargs)
        if channel is None:
            raise ValueError(f"{self.name}: channel is required")

        self.channel = channel
        self.add_requirement(channel, raw=False, seg=True)

    def apply_magnitude(self, mag) -> np.ndarray:
        raise NotImplementedError

    def generate(self, context: PatternContext):
        mask = context.segmentation(self.channel)

        labeled_mask = label(mask)
        props = regionprops(labeled_mask)

        if len(props) == 0:
            logger.warning(f"No props segmented: {self.experiment_name}")
            return np.zeros_like(mask)

        biggest_prop_area = 0
        for prop in props:
            if prop.area > biggest_prop_area:
                mask = labeled_mask == prop.label
                centroid = prop.centroid
                # the major axis as a (row, column) unit vector; skimage's
                # orientation is its angle from the row axis
                long_axis = (np.cos(prop.orientation), np.sin(prop.orientation))
                axis_length = prop.axis_major_length

                biggest_prop_area = prop.area

        rows, cols = np.meshgrid(
            np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing="ij"
        )

        mag = (rows - centroid[0]) * long_axis[0] + (cols - centroid[1]) * long_axis[1]
        mag = np.abs(mag) / (axis_length / 2)

        included = self.apply_magnitude(mag)

        return included * mask


class InnerPatternMethod(PatternAlongAxis):
    zoo_meta = ZooMeta(
        source="fly",
        title="Inner fraction",
        group="embryo",
        description="Applies a bar to the center of the long axis",
        kwargs={"channel": "638", "fraction": 0.1},
    )

    name = "embryo_inner"

    def __init__(self, fraction=0.1, **kwargs):
        """
        :param fraction: the band's extent, as a fraction (0-1) of the
            half-length of the long axis
        """
        super().__init__(**kwargs)

        if not 0 <= fraction <= 1:
            raise ValueError(
                f"{self.name}: fraction must be between 0 and 1, got {fraction}"
            )
        self.fraction = fraction

    def apply_magnitude(self, mag) -> np.ndarray:
        return mag < self.fraction


class OuterPatternMethod(PatternAlongAxis):
    zoo_meta = ZooMeta(
        source="fly",
        title="Outer fraction",
        group="embryo",
        description="Applies a bar to the poles of the long axis",
        kwargs={"channel": "638", "fraction": 0.1},
    )

    name = "embryo_outer"

    def __init__(self, fraction=0.1, **kwargs):
        """
        :param fraction: the band's extent, as a fraction (0-1) of the
            half-length of the long axis
        """
        super().__init__(**kwargs)

        if not 0 <= fraction <= 1:
            raise ValueError(
                f"{self.name}: fraction must be between 0 and 1, got {fraction}"
            )
        self.fraction = fraction

    def apply_magnitude(self, mag) -> np.ndarray:
        return mag > (1 - self.fraction)
