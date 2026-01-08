import numpy as np

from .pattern import PatternMethod, AcquiredImageRequest, DataDock
from skimage.measure import regionprops
import tifffile


class NucleusControlMethod(PatternMethod):

    name = "nucleus_control_base"

    def __init__(self, channel=None, **kwargs):
        super().__init__(**kwargs)

        if channel is None:
            raise AttributeError("NucleusControlMethod must be provided a "
                                 "segmentation channel in the .toml: e.g. nuc_channel = \"545\"")

        self.channel = channel
        self.add_requirement(channel, raw=True, seg=True)

    def process_prop(self, prop):
        return prop.image

    def generate(self, context) -> np.ndarray:

        seg = context.segmentation(self.channel)
        raw = context.raw(self.channel)

        h, w = self.pattern_shape

        new_img = np.zeros((int(h), int(w)), dtype=np.float16)

        for prop in regionprops(seg, intensity_image=raw):
            cell_stim = self.process_prop(prop)

            new_img[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] += cell_stim

        new_img_clamped = np.clip(new_img, 0, 1).astype(np.float16)

        return new_img_clamped


class BinaryNucleusClampModel(NucleusControlMethod):

    name = "binary_nucleus_clamp"

    def __init__(self, channel, clamp_target, **kwargs):

        super().__init__(channel=channel, **kwargs)

        self.clamp_target = clamp_target

    def process_prop(self, prop):
        if prop.intensity_mean > self.clamp_target:
            return prop.image * 0

        return prop.image


class CenteredImageModel(NucleusControlMethod):

    name = "centered_image"

    def __init__(self, channel="545", tif_path=None,
                 min_intensity=2000, max_intensity=5000, **kwargs):

        super().__init__(channel=channel, **kwargs)

        self.img = tifffile.imread(tif_path)
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

        self.target_image = None

    # todo: warp image to target size
    # def

    def make_target_image(self):

        img = self.img

        h, w = self.pattern_shape

        h = int(h)
        w = int(w)

        padding_h_up = (h - img.shape[0]) // 2
        padding_h_down = h - (padding_h_up + img.shape[0])
        padding_w_left = (w - img.shape[1]) // 2
        padding_w_right = w - (padding_w_left + img.shape[1])

        padded_img = np.pad(img, ((padding_h_up, padding_h_down), (padding_w_left, padding_w_right)))
        padded_img = np.array(padded_img, dtype=np.float16)

        max_v = np.max(img)
        min_v = np.min(img)

        padded_img = (padded_img - min_v) / (max_v - min_v)
        padded_img = np.clip(padded_img, 0, 1)

        padded_img = (padded_img * (self.max_intensity - self.min_intensity)) + self.min_intensity

        print(padded_img.shape)

        self.target_image = padded_img

    def get_target_intensity(self, prop):

        y, x = prop.centroid

        y = round(y)
        x = round(x)

        return self.target_image[y, x]

    def process_prop(self, prop):

        target = self.get_target_intensity(prop)

        if prop.intensity_mean > target:
            return prop.image * 0

        return prop.image

    def generate(self, context) -> np.ndarray:

        if self.target_image is None:
            self.make_target_image()

        return super().generate(context)


class GlobalCycleModel(NucleusControlMethod):

    name = "global_cycle"

    def __init__(self, channel, period_m=10, **kwargs):

        super().__init__(channel=channel, **kwargs)

        self.period_s = period_m * 60

    def generate(self, context) -> np.ndarray:

        t = context.time

        is_on = ((t // self.period_s) % 2) == 0

        h, w = self.pattern_shape

        return np.zeros((int(h), int(w))) * is_on
