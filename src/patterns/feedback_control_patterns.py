import numpy as np

from . import DataDock
from .pattern import PatternModel, AcquiredImageRequest
from skimage.measure import regionprops


class PerCellPatternModel(PatternModel):

    name = "per_cell_base"

    def __init__(self, experiment_name, camera_properties, channel=None):
        super().__init__(experiment_name, camera_properties)

        if channel is None:
            raise AttributeError(f"{experiment_name}: PerCellPatternModels must be provided a "
                                 f"segmentation channel in the .toml: e.g. channel = \"638\"")

        self.channel = channel
        self.seg_channel_id = None


    def initialize(self, experiment):
        super().initialize(experiment)

        channel = experiment.channels.get(self.channel, None)

        assert channel is not None, f"provided channel {self.channel} is not in experiment"

        self.seg_channel_id = channel.channel_id
        cell_seg_air = AcquiredImageRequest(channel.channel_id, False, True)

        return [cell_seg_air]

    def prop_vector(self, prop, vec):
        """
        :param prop: regionprops prop
        :param vec: (y, x) vector
        """
        bin_img = prop.image
        y = np.arange(bin_img.shape[0])
        x = np.arange(bin_img.shape[1])
        yy, xx = np.meshgrid(y, x)

        y_center, x_center = prop.centroid_local

        dot_prod = (yy - y_center) * vec[0] + (xx - x_center) * vec[1]

        return (dot_prod > 0) & bin_img

    def process_prop(self, prop) -> np.ndarray:

        return prop.image

    def generate(self, data_dock: DataDock) -> np.ndarray:

        seg: np.ndarray = data_dock.data[self.seg_channel_id]["seg"].data

        new_img = np.zeros(self.pattern_shape, dtype=np.float16)

        for prop in regionprops(seg):
            cell_stim = self.process_prop(prop)

            new_img[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] += cell_stim

        new_img_clamped = np.clip(new_img, 0, 1).astype(np.float16)

        return new_img_clamped

class SplitCenterModel
