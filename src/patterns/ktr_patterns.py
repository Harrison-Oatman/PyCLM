import numpy as np

from . import DataDock
from .pattern import PatternModel, AcquiredImageRequest
from skimage.measure import regionprops, regionprops_table


class KTRPatternModel(PatternModel):

    name = "per_cell_base"

    def __init__(self, experiment_name, camera_properties, nuc_channel=None, ktr_channel=None, **kwargs):
        super().__init__(experiment_name, camera_properties, **kwargs)

        if nuc_channel is None:
            raise AttributeError(f"{experiment_name}: PerCellPatternModels must be provided a "
                                 f"segmentation channel in the .toml: e.g. nuc_channel = \"545\"")

        if nuc_channel is None:
            raise AttributeError(f"{experiment_name}: PerCellPatternModels must be provided a "
                                 f"signal channel in the .toml: e.g. ktr_channel = \"638\"")

        self.nuc_channel = nuc_channel
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
        yy, xx = np.meshgrid(y, x, indexing="ij")

        y_center, x_center = prop.centroid_local

        dot_prod = (yy - y_center) * vec[0] + (xx - x_center) * vec[1]

        if self.gradient:
            framed = dot_prod * bin_img * (dot_prod > 0)
            out = framed / np.max(framed)

        else:
            out = (dot_prod > 0) * bin_img

        return out

    def process_prop(self, prop) -> np.ndarray:

        return prop.image

    def voronoi_rebuild(self, img):

        props_table = regionprops_table(img, properties=["centroid"])

        pts = np.stack([props_table["centroid-0"], props_table["centroid-1"]], axis=-1)

        kdtree = KDTree(pts)

        h, w = self.pattern_shape
        y = np.arange(h)
        x = np.arange(w)
        yy, xx = np.meshgrid(y, x, indexing="ij")

        query_pts = np.stack([yy.flatten(), xx.flatten()], axis=-1)

        _, nn = kdtree.query(query_pts, 1)

        out = np.reshape(nn, (int(h), int(w)))

        return out

    def generate(self, data_dock: DataDock) -> np.ndarray:

        seg: np.ndarray = data_dock.data[self.seg_channel_id]["seg"].data

        if self.voronoi:
            seg = self.voronoi_rebuild(seg)

        h, w = self.pattern_shape

        new_img = np.zeros((int(h), int(w)), dtype=np.float16)

        for prop in regionprops(seg):
            cell_stim = self.process_prop(prop)

            new_img[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] += cell_stim

        new_img_clamped = np.clip(new_img, 0, 1).astype(np.float16)

        return new_img_clamped


class NucleusControlModel(PatternModel):

    name = "nucleus_control_base"

    def __init__(self, experiment_name, camera_properties, nuc_channel=None, **kwargs):
        super().__init__(experiment_name, camera_properties, **kwargs)

        if nuc_channel is None:
            raise AttributeError(f"{experiment_name}: PerCellPatternModels must be provided a "
                                 f"segmentation channel in the .toml: e.g. nuc_channel = \"545\"")

        self.nuc_channel = nuc_channel
        self.seg_channel_id = None

    def initialize(self, experiment):
        super().initialize(experiment)

        channel = experiment.channels.get(self.nuc_channel, None)

        assert channel is not None, f"provided channel {self.nuc_channel} is not in experiment"

        self.seg_channel_id = channel.channel_id
        cell_seg_air = AcquiredImageRequest(channel.channel_id, True, True)

        return [cell_seg_air]

    def process_prop(self, prop):
        return prop.image

    def generate(self, data_dock: DataDock) -> np.ndarray:

        seg: np.ndarray = data_dock.data[self.seg_channel_id]["seg"].data
        raw: np.ndarray = data_dock.data[self.seg_channel_id]["raw"].data

        h, w = self.pattern_shape

        new_img = np.zeros((int(h), int(w)), dtype=np.float16)

        for prop in regionprops(seg, intensity_image=raw):
            cell_stim = self.process_prop(prop)

            new_img[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] += cell_stim

        new_img_clamped = np.clip(new_img, 0, 1).astype(np.float16)

        return new_img_clamped


class BinaryNucleusClampModel(NucleusControlModel):

    name = "binary_nucleus_clamp"

    def __init__(self, experiment_name, camera_properties, nuc_channel, clamp_target, **kwargs):

        super().__init__(experiment_name, camera_properties, nuc_channel, **kwargs)

        self.clamp_target = clamp_target

    def process_prop(self, prop):
        if prop.intensity_mean > self.clamp_target:
            return prop.image * 0

        return prop.image
