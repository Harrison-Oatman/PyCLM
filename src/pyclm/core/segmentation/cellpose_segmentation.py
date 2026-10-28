import logging

import numpy as np
from cv2 import resize

from .segmentation import *

logger = logging.getLogger(__name__)


class CellposeResource(SharedSegmentationResource):
    def __init__(self, pretrained_model="cpsam", gpu=True, **kwargs):
        super().__init__()

        self.pretrained_model = pretrained_model
        self.gpu = gpu

        logger.info(f"loading cellpose model: {pretrained_model} (gpu={gpu})")

        from cellpose import models

        self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    def eval(self, data, **cellpose_kwargs):
        """
        Masks for a 2D frame, or a ``(height, width, channel)`` image of up
        to three channels (Cellpose-SAM takes three; fewer are padded with
        empty channels, as Cellpose does).
        """
        from cellpose.transforms import convert_image

        data = np.asarray(data)
        channel_axis = -1 if data.ndim == 3 else None
        transformed_data = convert_image(data, channel_axis=channel_axis)
        cellpose_out = self.model.eval(
            transformed_data, channel_axis=-1, **cellpose_kwargs
        )

        return cellpose_out[0]


class CellposeResourceRequest(SharedSegmentationResourceRequest):
    def __init__(self, model_name, use_gpu):
        super().__init__(CellposeResource, pretrained_model=model_name, gpu=use_gpu)


class CellposeSegmentationMethod(SegmentationMethod):
    """
    Cellpose (default Cellpose-SAM, ``cpsam``) on one channel, or on up to
    three listed in the table's ``channels`` (a color image).

    Intensities are scaled to 0-1 before inference. By default with fixed
    camera counts, ``normlow`` to ``normhigh`` (a list gives one pair per
    channel); with ``auto_normalize = true`` from each frame's own
    percentiles, ``percentile_low`` to ``percentile_high`` per channel, which
    follows a field that brightens or bleaches.
    """

    name = "cellpose"
    max_channels = 3

    def __init__(
        self,
        experiment_name,
        model="cpsam",
        use_gpu=True,
        normlow=0,
        normhigh=5000,
        auto_normalize=False,
        percentile_low=1.0,
        percentile_high=99.0,
        **kwargs,
    ):
        """
        :param model: Cellpose pretrained model name or path
        :param use_gpu: run on the GPU
        :param normlow: camera count mapped to 0 (a list: one per channel)
        :param normhigh: camera count mapped to 1 (a list: one per channel)
        :param auto_normalize: scale each frame by its own percentiles instead
            of normlow / normhigh
        :param percentile_low: with auto_normalize, the percentile mapped to 0
        :param percentile_high: with auto_normalize, the percentile mapped to 1
        """
        super().__init__(experiment_name)

        self.model_name = model
        self.use_gpu = use_gpu
        self.auto_normalize = bool(auto_normalize)
        if not 0 <= float(percentile_low) < float(percentile_high) <= 100:
            raise ValueError(
                "percentile_low and percentile_high must satisfy "
                f"0 <= low < high <= 100 (got {percentile_low}, {percentile_high})"
            )
        self.percentiles = [float(percentile_low), float(percentile_high)]
        self.normlow = normlow
        self.normhigh = normhigh
        lists = [len(v) for v in (normlow, normhigh) if isinstance(v, list | tuple)]
        if len(set(lists)) > 1:
            raise ValueError(
                f"normlow and normhigh give {lists[0]} and {lists[1]} values; "
                "give one per listed channel"
            )

        self.cellpose_resource = None
        self.initialized = False

    def request_resource(self) -> Optional[SharedSegmentationResourceRequest]:
        return CellposeResourceRequest(self.model_name, self.use_gpu)

    def provide_resource(self, resource: SharedSegmentationResource):
        self.cellpose_resource = resource
        self.initialized = True

    def normalization_for(self, data) -> dict:
        """Cellpose's ``normalize`` argument for this frame's channel count."""
        data = np.asarray(data)
        n = data.shape[-1] if data.ndim == 3 else 1
        return normalization(
            self.auto_normalize,
            self.normlow,
            self.normhigh,
            self.percentiles[0],
            self.percentiles[1],
            n_channels=n,
        )

    def segment(self, data):
        assert self.initialized, "model was not yet provided a shared CellposeResource"

        masks = self.cellpose_resource.eval(
            data, normalize=self.normalization_for(data)
        )

        return masks


def normalization(
    auto_normalize, normlow, normhigh, percentile_low, percentile_high, n_channels=1
) -> dict:
    """
    Cellpose's ``normalize`` dictionary: per-frame percentiles, or fixed
    ``lowhigh`` bounds (one pair, or one per channel padded to the three
    channels Cellpose works in).
    """
    if auto_normalize:
        return {
            "normalize": True,
            "percentile": [float(percentile_low), float(percentile_high)],
        }
    lows = list(normlow) if isinstance(normlow, list | tuple) else None
    highs = list(normhigh) if isinstance(normhigh, list | tuple) else None
    if lows is None and highs is None:
        return {"lowhigh": [normlow, normhigh]}
    lows = lows or [normlow] * n_channels
    highs = highs or [normhigh] * n_channels
    if len(lows) != n_channels or len(highs) != n_channels:
        raise ValueError(
            f"normlow / normhigh give {len(lows)} / {len(highs)} values for "
            f"{n_channels} channel(s); give one per listed channel"
        )
    pairs = [[float(lo), float(hi)] for lo, hi in zip(lows, highs, strict=True)]
    # Cellpose pads an image to three channels; the padding is empty
    pairs += [[0.0, 1.0]] * (3 - len(pairs))
    return {"lowhigh": pairs}


class EmbryoSegmentationMethod(CellposeSegmentationMethod):
    name = "embryo_resizing"

    def __init__(
        self,
        experiment_name,
        model="embryomodel",
        use_gpu=True,
        normlow=0,
        normhigh=0.15,
        ideal_size=(100, 100),
        cache_result=True,
        **kwargs,
    ):
        super().__init__(experiment_name, model, use_gpu, normlow, normhigh, **kwargs)

        self.ideal_size = ideal_size
        self.cached_result = None
        self.do_cache = cache_result

    def segment(self, data):
        # use cached result if caching (e.g. because embryo doesn't move)
        if self.do_cache and (self.cached_result is not None):
            return self.cached_result

        # downscale to make the embryo smaller (cv2 sizes are (width, height))
        downscaled_frame = resize(np.asarray(data), self.ideal_size)

        out = super().segment(downscaled_frame)
        mask = out > 0

        # upscale back to full size
        height, width = np.asarray(data).shape[:2]
        big_mask = resize(mask.astype(float), (width, height)) > 0.5

        big_mask = np.array(big_mask)
        logger.debug(f"big_mask shape: {big_mask.shape}")

        if self.do_cache:
            self.cached_result = big_mask

        return big_mask
