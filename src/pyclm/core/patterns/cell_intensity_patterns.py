from typing import ClassVar

import numpy as np
import tifffile
from skimage.measure import regionprops

from .pattern import AcquiredImageRequest, DataDock, PatternMethod
from .zoo import ZooMeta


class NucleusControlMethod(PatternMethod):
    name = "nucleus_control_base"
    # early TOMLs named the channel nuc_channel (silently ignored until 2.0)
    renamed_kwargs: ClassVar[dict[str, str]] = {"nuc_channel": "channel"}

    def __init__(self, channel=None, tracks=False, **kwargs):
        """
        :param channel: preset of the nuclear channel to segment and measure
        :param tracks: run on the tracked labels (needs a ``[tracking]``
            table), so each nucleus keeps its id across timepoints
        """
        super().__init__(**kwargs)

        if channel is None:
            raise AttributeError(
                "NucleusControlMethod must be provided a "
                'segmentation channel in the .toml: e.g. channel = "545"'
            )

        self.channel = channel
        # with tracks = true the per-cell loop runs on tracked labels, so
        # prop.label is a track id that persists across timepoints
        self.use_tracks = bool(tracks)
        self.add_requirement(
            channel, raw=True, seg=not self.use_tracks, tracks=self.use_tracks
        )

    def process_prop(self, prop):
        return prop.image

    def cell_labels(self, context) -> np.ndarray:
        """The label image the per-cell loop runs on: tracked labels with ``tracks = true``, else the segmentation."""
        if self.use_tracks:
            return context.tracks(self.channel).labels
        return context.segmentation(self.channel)

    def generate(self, context) -> np.ndarray:
        seg = self.cell_labels(context)
        raw = context.raw(self.channel)

        h, w = self.pattern_shape

        new_img = np.zeros((int(h), int(w)), dtype=np.float16)

        for prop in regionprops(seg, intensity_image=raw):
            cell_stim = self.process_prop(prop)

            new_img[prop.bbox[0] : prop.bbox[2], prop.bbox[1] : prop.bbox[3]] += (
                cell_stim
            )

        new_img_clamped = np.clip(new_img, 0, 1).astype(np.float16)

        return new_img_clamped


class BinaryNucleusClampModel(NucleusControlMethod):
    name = "nucleus_clamp"
    renamed_kwargs: ClassVar[dict[str, str]] = {"clamp_target": "target"}

    def __init__(self, channel=None, target=10000, **kwargs):
        """
        :param channel: the channel whose nuclei are measured and lit
        :param target: mean nuclear intensity (camera counts) to hold each
            nucleus at: a nucleus dimmer than this is lit
        """
        super().__init__(channel=channel, **kwargs)

        self.target = target

    zoo_meta = ZooMeta(
        source="mcf10a",
        title="Bang-Bang Controller",
        kwargs={"channel": "545", "target": 65},
        description="Lights each nucleus whose mean intensity is below target.",
        group="per_nucleus",
    )

    def process_prop(self, prop):
        if prop.intensity_mean > self.target:
            return prop.image * 0

        return prop.image


class CenteredImageModel(NucleusControlMethod):
    name = "target_image"
    renamed_kwargs: ClassVar[dict[str, str]] = {
        "tif_path": "path",
        "min_intensity": "target_min",
        "max_intensity": "target_max",
    }
    zoo_meta = ZooMeta(
        source="mcf10a",
        title="Target Image",
        kwargs={
            "channel": "545",
            "path": "target_shield.tif",
            "target_min": 0,
            "target_max": 100,
        },
        description="Bang-bang control towards a target image centered on the field: each nucleus is lit while it is dimmer than the target at its centroid, rescaled to target_min..target_max.",
        group="per_nucleus",
    )

    def __init__(
        self,
        channel=None,
        path=None,
        target_min=2000,
        target_max=5000,
        **kwargs,
    ):
        """
        :param channel: the channel whose nuclei are measured and lit
        :param path: the target image (a TIF), centered on the field
        :param target_min: the target intensity (camera counts) of the
            image's darkest pixel
        :param target_max: the target intensity of its brightest pixel
        """
        super().__init__(channel=channel, **kwargs)

        if path is None:
            raise ValueError(f"{self.name}: path (the target TIF) is required")
        self.img = tifffile.imread(path)
        self.target_min = target_min
        self.target_max = target_max

        self.target_image = None

    def make_target_image(self):
        img = self.img

        h, w = self.pattern_shape

        h = int(h)
        w = int(w)

        padding_h_up = (h - img.shape[0]) // 2
        padding_h_down = h - (padding_h_up + img.shape[0])
        padding_w_left = (w - img.shape[1]) // 2
        padding_w_right = w - (padding_w_left + img.shape[1])

        padded_img = np.pad(
            img, ((padding_h_up, padding_h_down), (padding_w_left, padding_w_right))
        )
        padded_img = np.array(padded_img, dtype=np.float16)

        max_v = np.max(img)
        min_v = np.min(img)

        padded_img = (padded_img - min_v) / (max_v - min_v)
        padded_img = np.clip(padded_img, 0, 1)

        padded_img = (
            padded_img * (self.target_max - self.target_min)
        ) + self.target_min

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

    deprecated = (
        'use method = "global_pulse" with period_min = 2 * period_m, on_min = period_m'
    )
    zoo_meta = ZooMeta(
        same_as="global_pulse",
        equivalent="period_min = 2 * period_m, on_min = period_m",
    )

    def __init__(self, channel="545", period_m=10, **kwargs):
        super().__init__(channel=channel, **kwargs)

        self.period_s = period_m * 60

    def generate(self, context) -> np.ndarray:
        t = context.time

        is_on = ((t // self.period_s) % 2) == 0

        h, w = self.pattern_shape

        return np.ones((int(h), int(w))) * is_on
