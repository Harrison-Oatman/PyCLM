"""
Run embryo segmentation and end illumination experiment with PyCLM
Single pattern module using Cellpose segmentation + PCA-based end illumination
"""

from pyclm import run_pyclm, SegmentationMethod, PatternMethod
from pyclm.core.experiments import Experiment
from pyclm.core.patterns.feedback_control_patterns import PerCellPatternMethod, AcquiredImageRequest, DataDock
from pyclm.core.segmentation.cellpose_segmentation import CellposeSegmentationMethod

import numpy as np
from cellpose import models
from skimage import morphology, measure
from scipy import ndimage
from pathlib import Path
import tifffile
import logging
from skimage.transform import rescale, resize
import pickle
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# todo: add path where your masks will temporarily be saved
TMP_PATH = Path(r"")


def get_mask_path(experiment_name: str):

    parts = experiment_name.split(".")
    name = "_".join(parts[1:])
    path = TMP_PATH / f"{name}_masks.npy"

    return path


class BodySegmentationMethod(CellposeSegmentationMethod):
    """
    Image at the body plane of the embryo.
    Segments the body and passes to pattern method
    """

    def __init__(self, model="cpsam", use_gpu=True, normlow=0, normhigh=5000, rescale_factor=0.06, **kwargs):
        super().__init__(model, use_gpu, normlow, normhigh)

        self.rescale_factor = rescale_factor


    def segment(self, image):

        # shrink for cellpose
        image_rescaled = rescale(image, self.rescale_factor)

        masks_downscaled = super().segment(image_rescaled)

        # unshrink to return
        masks = np.round(resize(masks_downscaled, image.shape)).astype(int)

        # If multiple objects found, keep only the largest
        if masks.max() > 1:
            props = measure.regionprops(masks)
            largest = max(props, key=lambda r: r.area)
            binary_mask = masks == largest.label
            logger.debug(f"Cellpose found {masks.max()} objects, kept largest with area={largest.area}")
        else:
            binary_mask = masks > 0
            logger.debug(f"Cellpose found 1 object")

        return binary_mask


class BodyCapsPatternMethod(PatternMethod):
    """
    Takes in cap parameters
    Saves the cap mask and returns all zeros on pattern generation
    """

    def __init__(self,
                 experiment_name,
                 camera_properties,
                 channel=None,
                 coverage=10,
                 **kwargs):

        super().__init__(experiment_name, camera_properties)
        self.channel = channel
        self.seg_channel_id = None
        self.coverage = coverage

    def initialize(self, experiment):
        super().initialize(experiment)
        channel = experiment.channels.get(self.channel, None)
        assert channel is not None, f"provided channel {self.channel} is not in experiment"

        self.seg_channel_id = channel.channel_id

        return [AcquiredImageRequest(self.seg_channel_id, False, True)]

    # todo: fill in make caps function: input is embryo mask, output is caps mask
    def make_caps(self, binary_mask):
        coverage = self.coverage



    def generate(self, data: DataDock):
        binary_mask = data.data[self.seg_channel_id]["seg"].data

        caps = self.make_caps(binary_mask)

        caps_save_path = get_mask_path(self.experiment)

        np.save(caps_save_path, caps)

        return np.zeros(binary_mask.shape)


class NucleusSegmentation(CellposeSegmentationMethod):
    """
    Images at the histone plane.
    Output the segmented nuclei
    Uses cpsam

    Nothing needs to be implemented here.
    """

    name = "segment_nuclei"


class NuclearCycleAppliesCap(PatternMethod):
    """
    Loads body cap from file if it exists yet
    Detects nuclear cycle, and determines whether to start running the pattern
    If not, all zeros
    if yet, applies the mask
    """

    def __init__(self,
                 experiment_name,
                 camera_properties,
                 channel=None,
                 starting_nc = 11,
                 pkl_file = None, **kwargs
                 ):

        super().__init__(experiment_name, camera_properties)
        self.channel = channel
        self.seg_channel_id = None

        self.starting_nc = starting_nc
        self.pkl_file = pkl_file

        with open(pkl_file, "rb") as f:
            self.forest_model: RandomForestClassifier = pickle.load(f)

        self.mask = None

        self.all_predictions = []

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:

        super().initialize(experiment)

        channel = experiment.channels.get(self.channel, None)
        assert channel is not None, f"provided channel {self.channel} is not in experiment"

        self.seg_channel_id = channel.channel_id

        return [AcquiredImageRequest(self.seg_channel_id, False, True)]

    # todo: add your prediction code here
    def predict_nc_cycle(self, nucleus_mask) -> str:
        classifier = self.forest_model

        # uyour code

    # todo: write code that will return true or false whether to start running the pattern
    def choose_to_apply(self) -> bool:

        predicted_ncs = self.all_predictions

        # your code

    def check_for_mask(self):

        mask_path = get_mask_path(self.experiment)

        if not mask_path.exists():
            return False

        self.mask = np.load(mask_path).astype(np.float16)

        return True

    def generate(self, data_dock: DataDock) -> np.ndarray:

        nucleus_mask = data_dock.data[self.seg_channel_id]["seg"].data

        predicted_nuclear_cycle = self.predict_nc_cycle(nucleus_mask)

        self.all_predictions.append(predicted_nuclear_cycle)

        apply_pattern = self.choose_to_apply()

        if not self.mask:
            self.check_for_mask()

        if apply_pattern:
            if self.mask:
                return self.mask

        return np.zeros(nucleus_mask.shape)

# todo: make body.toml
#    segmentation method: body_segmentation
#       needs norm_high to be speicified as keyword (maximum intensity for normalization)
#    pattern method: make_caps
#       needs coverage to be specified as keyword
#    every_t like 1000 (for imaging and stimulation)
#
# todo: make nc.toml
#   segmentation method: nucleus_segmentation
#       needs norm_high to be speicified as keyword (maximum intensity for normalization)
#   pattern method: apply_caps
#       needs pkl_path to be sepcified

def main():
    """
    Run PyCLM experiment with embryo end illumination (all-in-one pattern module)
    """

    # Set working directory (where your TOML files are)
    working_directory = Path(r"Z:\Microscopy Data\Matteo\Embryo Illumination\Basic Seg Illumination\run2_cellpose")

    # Register custom pattern method
    segmentation_methods = {
        "body_segmentation": BodySegmentationMethod,
        "nucleus_segmentation": NucleusSegmentation,
    }

    pattern_methods = {
        "make_caps": BodyCapsPatternMethod,
        "apply_caps": NuclearCycleAppliesCap,
    }

    print("=" * 70)
    print("STARTING EMBRYO END ILLUMINATION EXPERIMENT (PCA Method)")
    print("=" * 70)
    print(f"Working directory: {working_directory}")
    print(f"Debug output: Z:\\Microscopy Data\\Matteo\\Embryo Illumination\\Debug Output")
    print("=" * 70)

    # Run PyCLM
    run_pyclm(
        working_directory,
        segmentation_methods=segmentation_methods,
        pattern_methods=pattern_methods
    )

if __name__ == "__main__":
    main()
