"""
Run embryo segmentation and end illumination experiment with PyCLM
Single pattern module that handles both segmentation and illumination using PCA
"""

from pyclm import run_pyclm
from pyclm.core.patterns.feedback_control_patterns import PerCellPatternMethod, AcquiredImageRequest, DataDock

import numpy as np
from skimage import filters, morphology, measure
from scipy import ndimage
from pathlib import Path
import tifffile
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PATTERN MODULE - HANDLES EVERYTHING (SEGMENTATION + ILLUMINATION)
# ============================================================================

class EmbryoEndPatternPCA(PerCellPatternMethod):
    """
    All-in-one pattern module:
    - Segments embryo from raw image
    - Uses PCA to find embryo orientation
    - Creates filled cap illumination at both ends
    - Saves debug TIFFs for troubleshooting
    """

    name = "embryo_ends_pca"

    def __init__(
        self,
        experiment_name,
        camera_properties,
        channel=None,
        coverage=30,
        min_area=10000,
        smoothing_sigma=2,
        save_debug=True,
        debug_path=None,
        **kwargs
    ):
        super().__init__(experiment_name, camera_properties, channel, **kwargs)

        self.coverage = coverage  # percentage of major axis to illuminate from each end
        self.min_area = min_area
        self.smoothing_sigma = smoothing_sigma
        self.save_debug = save_debug

        # Setup debug output directory
        if debug_path is None:
            self.debug_path = Path(r"Z:\Microscopy Data\Matteo\Embryo Illumination\Debug Output")
        else:
            self.debug_path = Path(debug_path)

        self.debug_path.mkdir(parents=True, exist_ok=True)
        self.frame_counter = 0

        logger.info(f"Initialized {self.name} with {coverage}% coverage - debug_path: {self.debug_path}")

    def initialize(self, experiment):
        super().initialize(experiment)

        channel = experiment.channels.get(self.channel, None)
        assert channel is not None, f"provided channel {self.channel} is not in experiment"

        self.seg_channel_id = channel.channel_id

        # Request raw image only - we do everything ourselves
        raw_image_request = AcquiredImageRequest(channel.channel_id, needs_raw=True, needs_seg=False)

        return [raw_image_request]

    def segment_embryo(self, image):
        """
        Segment embryo from raw image using Otsu thresholding
        """
        # Smooth image
        smoothed = filters.gaussian(image, sigma=self.smoothing_sigma)

        # Otsu threshold
        thresh = filters.threshold_otsu(smoothed)
        binary = smoothed > thresh

        # Clean up: remove small objects, fill holes
        binary = morphology.remove_small_objects(binary, min_size=self.min_area)
        binary = ndimage.binary_fill_holes(binary)

        # Keep only largest component (the embryo)
        labeled = measure.label(binary)
        if labeled.max() == 0:
            return binary.astype(np.uint8)

        regions = measure.regionprops(labeled)
        largest_region = max(regions, key=lambda r: r.area)
        binary = labeled == largest_region.label

        return binary.astype(np.uint8)

    @staticmethod
    def find_embryo_ends_pca(binary_mask):
        """
        Find embryo ends using PCA (Principal Component Analysis)

        Returns:
        --------
        coords : np.ndarray
            All embryo pixel coordinates (y, x)
        center : np.ndarray
            Center coordinates
        major_axis : np.ndarray
            Unit vector along major axis
        projections : np.ndarray
            Projection of each pixel onto major axis
        """
        # Get coordinates of all embryo pixels
        coords = np.argwhere(binary_mask)

        if len(coords) == 0:
            return coords, np.array([0, 0]), np.array([0, 0]), np.array([])

        # Center the coordinates
        center = coords.mean(axis=0)
        centered_coords = coords - center

        # Compute covariance matrix
        cov_matrix = np.cov(centered_coords.T)

        # Get eigenvectors (principal components)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # The eigenvector with largest eigenvalue is the major axis
        major_axis_idx = np.argmax(eigenvalues)
        major_axis = eigenvectors[:, major_axis_idx].real

        # Project all points onto the major axis
        projections = np.dot(centered_coords, major_axis)

        return coords, center, major_axis, projections

    @staticmethod
    def create_end_caps(coords, center, major_axis, projections, illuminate_distance, shape):
        """
        Create filled cap masks at both ends using convex hull

        Returns:
        --------
        caps_mask : np.ndarray
            Binary mask with both end caps filled
        """
        end1_cap_mask = np.zeros(shape, dtype=bool)
        end2_cap_mask = np.zeros(shape, dtype=bool)

        min_projection = projections.min()
        max_projection = projections.max()

        # Assign pixels to caps based on distance along major axis
        for i, coord in enumerate(coords):
            projection = projections[i]
            dist_from_end1 = projection - min_projection
            dist_from_end2 = max_projection - projection

            y, x = coord

            # Assign to closer end
            if dist_from_end1 <= illuminate_distance and dist_from_end1 < dist_from_end2:
                end1_cap_mask[y, x] = True
            elif dist_from_end2 <= illuminate_distance:
                end2_cap_mask[y, x] = True

        # Fill caps using convex hull to create solid regions
        if end1_cap_mask.any():
            end1_cap_mask = morphology.convex_hull_image(end1_cap_mask)
            # Constrain to region near end1
            for y, x in np.argwhere(end1_cap_mask):
                centered = np.array([y, x]) - center
                proj = np.dot(centered, major_axis)
                if proj - min_projection > illuminate_distance:
                    end1_cap_mask[y, x] = False

        if end2_cap_mask.any():
            end2_cap_mask = morphology.convex_hull_image(end2_cap_mask)
            # Constrain to region near end2
            for y, x in np.argwhere(end2_cap_mask):
                centered = np.array([y, x]) - center
                proj = np.dot(centered, major_axis)
                if max_projection - proj > illuminate_distance:
                    end2_cap_mask[y, x] = False

        # Combine both caps
        caps_mask = end1_cap_mask | end2_cap_mask

        return caps_mask

    def generate(self, data_dock: DataDock):
        """
        Main processing: segment embryo, find ends with PCA, create illumination pattern
        """
        try:
            # Get raw image
            raw_image = data_dock.data[self.seg_channel_id]["raw"].data

            # Step 1: Segment the embryo
            binary_mask = self.segment_embryo(raw_image)

            if not binary_mask.any():
                logger.warning("No embryo found in image")
                if self.save_debug:
                    self._save_debug_tiffs(raw_image, binary_mask, np.zeros_like(raw_image))
                return np.zeros_like(raw_image, dtype=np.float32)

            # Step 2: Use PCA to find embryo orientation and ends
            coords, center, major_axis, projections = self.find_embryo_ends_pca(binary_mask)

            if len(coords) == 0:
                logger.warning("No embryo pixels found")
                if self.save_debug:
                    self._save_debug_tiffs(raw_image, binary_mask, np.zeros_like(raw_image))
                return np.zeros_like(raw_image, dtype=np.float32)

            # Calculate embryo length along major axis
            embryo_length = projections.max() - projections.min()

            # Calculate distance to illuminate from each end
            illuminate_distance = (self.coverage / 100) * embryo_length

            # Step 3: Create filled cap masks at both ends
            pattern_mask = self.create_end_caps(
                coords, center, major_axis, projections,
                illuminate_distance, binary_mask.shape
            )

            # Calculate angle for logging
            angle_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

            logger.info(f"Created pattern: "
                       f"length={embryo_length:.1f}px, "
                       f"angle={angle_deg:.1f}°, "
                       f"illuminated={pattern_mask.sum():.0f}px")

            # Save debug TIFFs
            if self.save_debug:
                self._save_debug_tiffs(raw_image, binary_mask, pattern_mask)

            self.frame_counter += 1

            return pattern_mask.astype(np.float32)

        except Exception as e:
            logger.error(f"Error generating pattern: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.pattern_shape, dtype=np.float32)

    def _save_debug_tiffs(self, raw_image, segmented_mask, pattern_mask):
        """
        Save debug TIFFs for troubleshooting:
        1. Raw image
        2. Segmentation mask
        3. Pattern mask
        4. Overlay (raw + pattern highlighted)
        """
        try:
            # 1. Save raw image
            raw_path = self.debug_path / f"frame_{self.frame_counter:04d}_1_raw.tif"
            tifffile.imwrite(raw_path, raw_image.astype(np.uint16))

            # 2. Save segmentation mask
            seg_path = self.debug_path / f"frame_{self.frame_counter:04d}_2_segmented.tif"
            tifffile.imwrite(seg_path, (segmented_mask * 255).astype(np.uint8))

            # 3. Save pattern mask
            pattern_path = self.debug_path / f"frame_{self.frame_counter:04d}_3_pattern.tif"
            tifffile.imwrite(pattern_path, (pattern_mask * 255).astype(np.uint8))

            # 4. Save overlay (raw image with pattern highlighted in red)
            overlay = raw_image.copy().astype(np.float32)
            overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
            overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
            overlay_rgb[pattern_mask > 0, 0] = 1.0  # Red channel
            overlay_rgb[pattern_mask > 0, 1] = 0.0  # Green channel
            overlay_rgb[pattern_mask > 0, 2] = 0.0  # Blue channel

            overlay_path = self.debug_path / f"frame_{self.frame_counter:04d}_4_overlay.tif"
            tifffile.imwrite(overlay_path, (overlay_rgb * 255).astype(np.uint8))

            logger.debug(f"Saved debug TIFFs for frame {self.frame_counter}")

        except Exception as e:
            logger.error(f"Error saving debug TIFFs: {e}")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """
    Run PyCLM experiment with embryo end illumination (all-in-one pattern module)
    """

    # Set working directory (where your TOML files are)
    working_directory = Path(r"Z:\Microscopy Data\Matteo\Embryo Illumination\Basic Seg Illumination\run1")

    # Register custom pattern method
    pattern_methods = {
        "embryo_ends_pca": EmbryoEndPatternPCA,
    }

    print("="*70)
    print("STARTING EMBRYO END ILLUMINATION EXPERIMENT (PCA Method)")
    print("="*70)
    print(f"Working directory: {working_directory}")
    print(f"Debug output: Z:\\Microscopy Data\\Matteo\\Embryo Illumination\\Debug Output")
    print("="*70)

    # Run PyCLM
    run_pyclm(
        working_directory,
        pattern_methods=pattern_methods
    )


if __name__ == "__main__":
    main()
