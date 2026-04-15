from cv2 import resize

from .segmentation import *


class CellposeResource(SharedSegmentationResource):
    def __init__(self, pretrained_model="cpsam", gpu=True, **kwargs):
        super().__init__()

        self.pretrained_model = pretrained_model
        self.gpu = gpu

        print(f"requesting cellpose model: {pretrained_model}")

        from cellpose import models

        self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    def eval(self, data, **cellpose_kwargs):
        from cellpose.transforms import convert_image

        transformed_data = convert_image(data)
        cellpose_out = self.model.eval(transformed_data, **cellpose_kwargs)

        return cellpose_out[0]


class CellposeResourceRequest(SharedSegmentationResourceRequest):
    def __init__(self, model_name, use_gpu):
        super().__init__(CellposeResource, pretrained_model=model_name, gpu=use_gpu)


class CellposeSegmentationMethod(SegmentationMethod):
    name = "cellpose"

    def __init__(
        self,
        experiment_name,
        model="cpsam",
        use_gpu=True,
        normlow=0,
        normhigh=5000,
        **kwargs,
    ):
        super().__init__(experiment_name)

        print(model)

        self.model_name = model
        self.use_gpu = use_gpu
        self.normalization = {"lowhigh": [normlow, normhigh]}

        self.cellpose_resource = None
        self.initialized = False

    def request_resource(self) -> Optional[SharedSegmentationResourceRequest]:
        return CellposeResourceRequest(self.model_name, self.use_gpu)

    def provide_resource(self, resource: SharedSegmentationResource):
        self.cellpose_resource = resource
        self.initialized = True

    def segment(self, data):
        assert self.initialized, "model was not yet provided a shared CellposeResource"

        masks = self.cellpose_resource.eval(data, normalize=self.normalization)

        return masks


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
        super().__init__(experiment_name, model, use_gpu, normlow, normhigh)

        self.ideal_size = ideal_size
        self.cached_result = None
        self.do_cache = cache_result

    def segment(self, data):
        # use cached result if caching (e.g. because embryo doesn't move)
        if self.do_cache and self.cached_result:
            return self.cached_result

        # downscale to make the embryo smaller
        downscaled_frame = resize(data, self.ideal_size)

        out = super().segment(downscaled_frame)
        mask = out > 0

        # upscale back to full size
        big_mask = resize(mask.astype(float), data.shape, order=1) > 0.5

        if self.do_cache:
            self.cached_result = big_mask

        return big_mask
