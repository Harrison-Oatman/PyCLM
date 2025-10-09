from pyclm import run_pyclm, SegmentationMethod
from pyclm.core.patterns.feedback_control_patterns import PerCellPatternMethod, AcquiredImageRequest, DataDock

import numpy as np
from skimage import filters, morphology, measure
from scipy import ndimage

from pathlib import Path
import tifffile
import math


class EllipseBoxPattern(PerCellPatternMethod):
    """
    For each segmented embryo:
    - Segment the embryo from raw image
    - Fit ellipse (via regionprops)
    - Find endpoints of the major axis
    - Draw solid boxes of light at those endpoints
    """

    name = "ellipse_box"

    def __init__(
        self,
        experiment_name,
        camera_properties,
        channel=None,
        coverage=15,
        min_area=1000,
        smoothing_sigma=2,
        **kwargs
    ):
        super().__init__(experiment_name, camera_properties, channel, **kwargs)
        self.coverage = coverage  # size of the square boxes in pixels (as % of major axis)
        self.min_area = min_area  # minimum area for valid embryo region
        self.smoothing_sigma = smoothing_sigma  # gaussian smoothing parameter

    def initialize(self, experiment):
        super().initialize(experiment)

        channel = experiment.channels.get(self.channel, None)
        assert channel is not None, f"provided channel {self.channel} is not in experiment"

        self.seg_channel_id = channel.channel_id

        # Request raw (non-segmented) image
        raw_image_request = AcquiredImageRequest(channel.channel_id, True, False)

        return [raw_image_request]

    def segment_embryo(self, image):
        """
        Simple binary segmentation to isolate fly embryo.

        Parameters:
        -----------
        image : np.ndarray
            Input grayscale image

        Returns:
        --------
        binary_mask : np.ndarray
            Binary mask with embryo segmented
        """
        # Smooth image to reduce noise
        smoothed = filters.gaussian(image, sigma=self.smoothing_sigma)

        # Otsu threshold for binary segmentation
        thresh = filters.threshold_otsu(smoothed)
        binary = smoothed > thresh

        # Clean up: remove small objects, fill holes
        binary = morphology.remove_small_objects(binary, min_size=self.min_area)
        binary = ndimage.binary_fill_holes(binary)

        # Keep only largest connected component (the embryo)
        labeled = measure.label(binary)
        if labeled.max() == 0:
            return binary.astype(np.uint8)

        regions = measure.regionprops(labeled)
        largest_region = max(regions, key=lambda r: r.area)
        binary = labeled == largest_region.label

        base_path = Path(r"E:\Matteo\2025_10_09 test_ellipses\outputs")
        tifffile.imwrite(base_path / f"{self.experiment}_raw.tif", image)
        tifffile.imwrite(base_path / f"{self.experiment}_segmented.tif", binary)

        return binary.astype(np.uint8)

    @staticmethod
    def draw_oriented_box(mask, cx, cy, major, minor, angle, coverage):
        """
        I need to change the function name to illuminate ends of ellipse
        :param mask:
        :param cx:
        :param cy:
        :param major:
        :param minor:
        :param angle:
        :param coverage:
        :return:
        """

        h, w = mask.shape[:2]
        y, x = np.mgrid[0:h, 0:w]

        # shift coordinates relative to ellipse center
        x_rel = x - cx
        y_rel = y - cy

        # Rotation (align coordinate frame with ellipse)
        angle = -angle # corrects mirroring
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = -x_rel * cos_a + y_rel * sin_a # along major axis
        y_rot = x_rel * cos_a + y_rel * sin_a  # along minor axis

        # Half lengths
        a = major/2
        b = minor/2

        # Ellipse equation (inside = True)
        inside_ellipse = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1.0

        # Cut off for illuminated ends
        cutoff = a * (1 - coverage / 100)
        illuminated = (x_rot >= cutoff) | (x_rot <= -cutoff)

        # Combine both conditions
        #mask = inside_ellipse & illuminated
        mask = inside_ellipse

        return mask

    def process_prop(self, prop):
        """
        Process a single regionprop to create illumination pattern.

        Parameters:
        -----------
        prop : RegionProperties
            Regionprop from skimage.measure.regionprops

        Returns:
        --------
        mask : np.ndarray
            Pattern mask with boxes at ellipse endpoints
        """
        mask = np.zeros_like(prop.image, dtype=np.float32)

        y0, x0 = prop.centroid_local
        major = prop.major_axis_length
        minor = prop.minor_axis_length
        orientation = prop.orientation
        coverage = self.coverage

        # Illuminate ellipse ends
        mask = self.draw_oriented_box(mask, x0, y0, major, minor, orientation, coverage)

        return mask

    def generate(self, data_dock: DataDock):
        """
        Main processing method: segment embryo and create pattern.

        Parameters:
        -----------
        data_dock : DataDock
            Data access object from pyclm

        Returns:
        --------
        pattern_mask : np.ndarray
            Full-size pattern mask for illumination
        """
        # Get raw image
        raw_image = data_dock.data[self.seg_channel_id]["raw"].data

        # Segment embryo
        binary_mask = self.segment_embryo(raw_image)

        # Get regionprops from segmented mask
        labeled = measure.label(binary_mask)
        props = measure.regionprops(labeled)

        # If no embryo found, return empty pattern
        if len(props) == 0:
            return np.zeros_like(raw_image, dtype=np.float32)

        # Create full-size pattern mask
        pattern_mask = np.zeros_like(raw_image, dtype=np.float32)

        # Process each region (typically should be just one embryo)
        for prop in props:
            # Get local pattern for this region
            local_mask = self.process_prop(prop)

            # Place into full image using bounding box
            min_row, min_col, max_row, max_col = prop.bbox
            pattern_mask[min_row:max_row, min_col:max_col] = local_mask

        base_path = Path(r"E:\Matteo\2025_10_09 test_ellipses\outputs")
        tifffile.imwrite(base_path / f"{self.experiment}_pattern.tif", pattern_mask)

        return pattern_mask

def main():

    working_directory = Path(r"E:\Matteo\2025_10_09 test_ellipses")

    pattern_methods = {
        "ellipse_box": EllipseBoxPattern,
    }

    run_pyclm(working_directory, pattern_methods=pattern_methods)

if __name__ == "__main__":
    main()
