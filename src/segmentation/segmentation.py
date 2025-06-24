import numpy as np
from typing import Optional
from uuid import uuid4


class SharedSegmentationResource:

    def __init__(self, **kwargs):
        pass


class SharedSegmentationResourceRequest:

    """
    Used to avoid loading duplicates of large models.
    """

    def __init__(self, resource: type[SharedSegmentationResource], init_kwargs: dict):
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


class SegmentationModel:
    name = "base_model"

    def __init__(self, experiment_name, **kwargs):
        pass

    def request_resource(self) -> Optional[SharedSegmentationResourceRequest]:
        pass

    def provide_resource(self, resource: SharedSegmentationResource):
        pass

    def segment(self, data: np.ndarray) -> np.ndarray:

        raise NotImplementedError("base_model does not implement segment")




