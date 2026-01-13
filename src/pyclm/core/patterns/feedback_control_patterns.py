import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from skimage.measure import label, regionprops, regionprops_table

from .pattern import AcquiredImageRequest, DataDock, PatternMethod


class PerCellPatternMethod(PatternMethod):
    """
    Base class for the per-cell pattern methods. This class provides the basic structure for generating patterns
    based on the properties of segmented cells. Subclasses should implement the `process_prop` method to define
    how each cell's properties are used to generate the pattern, and include any additional parameters needed for the
    specific pattern in their `__init__` method.
    """

    name = "per_cell_base"

    def __init__(
        self, channel=None, voronoi=False, gradient=False, direction=1, **kwargs
    ):
        super().__init__(**kwargs)

        if channel is None:
            raise AttributeError(
                "PerCellPatternModels must be provided a "
                'segmentation channel in the .toml: e.g. channel = "638"'
            )

        self.gradient = gradient
        self.voronoi = voronoi
        self.direction = direction

        self.channel = channel

        # request the segmentation of the provided channel, to be used by self.generate.
        # This will ensure that the data is available when generate is called.
        self.add_requirement(channel_name=channel, seg=True)

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

    def generate(self, context) -> np.ndarray:
        seg = context.segmentation(self.channel)

        if self.voronoi:
            px_dis = distance_transform_edt(seg == 0)
            seg = self.voronoi_rebuild(seg)

            seg = seg * (px_dis < 50)

        h, w = self.pattern_shape

        new_img = np.zeros((int(h), int(w)), dtype=np.float16)

        for prop in regionprops(seg):
            cell_stim = self.process_prop(prop)

            new_img[prop.bbox[0] : prop.bbox[2], prop.bbox[1] : prop.bbox[3]] += (
                cell_stim
            )

        new_img_clamped = np.clip(new_img, 0, 1).astype(np.float16)

        return new_img_clamped


class RotateCcwModel(PerCellPatternMethod):
    name = "rotate_ccw"

    def __init__(self, channel=None, **kwargs):
        super().__init__(channel=channel, **kwargs)

    def process_prop(self, prop) -> np.ndarray:
        center_y, center_x = self.pattern_shape[0] / 2, self.pattern_shape[1] / 2
        prop_centroid = prop.centroid

        vec = -(prop_centroid[1] - center_x), prop_centroid[0] - center_y

        return self.prop_vector(prop, vec)


class MoveOutModel(PerCellPatternMethod):
    name = "move_out"

    def __init__(self, channel=None, **kwargs):
        super().__init__(channel=channel, **kwargs)

    def process_prop(self, prop) -> np.ndarray:
        center_y, center_x = self.pattern_shape[0] / 2, self.pattern_shape[1] / 2
        prop_centroid = prop.centroid

        vec = prop_centroid[0] - center_y, (prop_centroid[1] - center_x)

        return self.prop_vector(prop, vec)


class MoveInModel(PerCellPatternMethod):
    name = "move_in"

    def __init__(self, channel=None, **kwargs):
        super().__init__(channel=channel, **kwargs)

    def process_prop(self, prop) -> np.ndarray:
        center_y, center_x = self.pattern_shape[0] / 2, self.pattern_shape[1] / 2
        prop_centroid = prop.centroid

        vec = -(prop_centroid[0] - center_y), -(prop_centroid[1] - center_x)

        return self.prop_vector(prop, vec)


class MoveDownModel(PerCellPatternMethod):
    name = "move_down"

    def __init__(self, channel=None, **kwargs):
        super().__init__(channel=channel, **kwargs)

    def process_prop(self, prop) -> np.ndarray:
        center_y, center_x = self.pattern_shape[0] / 2, self.pattern_shape[1] / 2
        prop_centroid = prop.centroid

        vec = (1, 0)

        return self.prop_vector(prop, vec)


class BounceModel(PerCellPatternMethod):
    name = "fb_bounce"

    def __init__(self, channel=None, t_loop=60, **kwargs):
        self.t_loop_s = t_loop * 60
        self.down = True

        super().__init__(channel=channel, **kwargs)

    def process_prop(self, prop) -> np.ndarray:
        center_y, center_x = self.pattern_shape[0] / 2, self.pattern_shape[1] / 2
        prop_centroid = prop.centroid

        vec = (1, 0) if self.down else (-1, 0)

        return self.prop_vector(prop, vec)

    def generate(self, context) -> np.ndarray:
        t = context.time
        t = t % self.t_loop_s

        halfway = self.t_loop_s / 2

        if t > halfway:
            self.down = False

        return super().generate(context)


class DensityModel(PerCellPatternMethod):
    name = "density_gradient"

    def __init__(self, channel=None, **kwargs):
        super().__init__(channel=channel, **kwargs)

    def generate(self, context) -> np.ndarray:
        seg = context.segmentation(self.channel)

        if self.voronoi:
            px_dis = distance_transform_edt(seg == 0)
            seg = self.voronoi_rebuild(seg)

            seg = seg * (px_dis < 50)

        h, w = self.pattern_shape

        new_img = np.zeros((int(h), int(w)), dtype=np.float16)

        density = generate_density(seg)

        if self.direction == -1:
            dy, dx = np.gradient(density)
        else:
            dy, dx = np.negative(np.gradient(density))

        grad_direction = np.arctan2(dy, dx)

        for prop in regionprops(seg):
            prop_centroid = np.round(prop.centroid).astype(int)

            vec = grad_direction[prop_centroid[0], prop_centroid[1]]
            vec = (np.sin(vec), np.cos(vec))

            cell_stim = self.prop_vector(prop, vec)

            new_img[prop.bbox[0] : prop.bbox[2], prop.bbox[1] : prop.bbox[3]] += (
                cell_stim
            )

        new_img_clamped = np.clip(new_img, 0, 1).astype(np.float16)

        return new_img_clamped
