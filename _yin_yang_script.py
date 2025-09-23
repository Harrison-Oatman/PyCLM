from pyclm import PatternMethod
from skimage.io import imread
import numpy as np
from pathlib import Path

from pyclm.src.experiments import Experiment
from pyclm.src.patterns import AcquiredImageRequest
from main import run_pyclm


class StaticImage(PatternMethod):

    def __init__(self,  experiment_name, camera_properties, image_path=None, **kwargs):
        super().__init__(experiment_name, camera_properties)

        img = imread(image_path, as_gray=True).astype(float)
        img = np.pad(img, 100, constant_values=0.)

        self.img = img / np.max(img)

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        super().initialize(experiment)

        assert self.img.shape == self.pattern_shape, (f"image provided is {self.img.shape}, "
                                                      f"but must be {self.pattern_shape}")

        return []

    def generate(self, data):

        return self.img


def main():

    # base_path = Path(r"E:\Yang\20250918 PyCLM 2 color patterning\round_2")
    config_path = Path(r"C:\Users\Nikon\Desktop\Code\FeedbackControl\pyclm_config.toml")
    #
    # pattern_methods = {"image": StaticImage}
    #
    # run_pyclm(base_path, config_path, pattern_methods=pattern_methods)

    next_path = Path(r"E:\Yang\20250918 PyCLM 2 color patterning\imaging")
    run_pyclm(next_path, config_path)


if __name__ == "__main__":
    main()
