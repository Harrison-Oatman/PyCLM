import numpy as np


class SegmentationModel:
    name = "base_model"

    def __init__(self, experiment_name, **kwargs):
        pass

    def segment(self, data: np.ndarray) -> np.ndarray:

        raise NotImplementedError("base_model does not implement segment")


