from typing import Optional
from uuid import uuid4

import numpy as np


class SharedSegmentationResource:
    def __init__(self, **kwargs):
        pass


class SharedSegmentationResourceRequest:
    """
    Used to avoid loading duplicates of large models.
    """

    def __init__(self, resource: type[SharedSegmentationResource], **init_kwargs):
        """

        :param resource: **class** of shared resource
        :param init_kwargs: kwargs for initializing the shared resource
        """

        self.resource = resource  # this is an uninitialized resource
        self.init_kwargs = init_kwargs
        self.request_id = uuid4()

    def __eq__(self, other):
        same_resource_type = self.resource == other.resource
        same_kwargs = self.init_kwargs == other.init_kwargs

        return same_resource_type & same_kwargs


class SegmentationMethod:
    """
    Turns a frame into a label image. ``segment`` receives a 2D frame, or,
    when the ``[segmentation]`` table lists several ``channels``, one
    ``(height, width, channel)`` image with the channels in that order.
    ``max_channels`` is how many channels a method accepts (1: a method that
    only knows 2D frames); ``pyclm check`` refuses a table that lists more.
    """

    name = "base_model"
    max_channels: int = 1

    def __init__(self, experiment_name, **kwargs):
        self.experiment_name = experiment_name

    def request_resource(self) -> SharedSegmentationResourceRequest | None:
        pass

    def provide_resource(self, resource: SharedSegmentationResource):
        pass

    def segment(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("base_model does not implement segment")
