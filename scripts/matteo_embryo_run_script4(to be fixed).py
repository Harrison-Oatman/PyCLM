"""
Dual Experiment: Body Segmentation + NC-Triggered Illumination
Uses the framework you provided with all sections filled in
"""

from pyclm import run_pyclm, SegmentationMethod, PatternMethod
from pyclm.core.experiments import Experiment
from pyclm.core.patterns.feedback_control_patterns import PerCellPatternMethod, AcquiredImageRequest, DataDock
from pyclm.core.segmentation.cellpose_segmentation import CellposeSegmentationMethod

import numpy as np
from cellpose import models
from skimage import morphology, measure
from scipy import ndimage, spatial
from pathlib import Path
import tifffile
import logging
from skimage.transform import rescale, resize
import pickle
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# Path where caps masks will be temporarily be saved (shared between experiments)
TMP_PATH = Path(r"Z:\Microscopy Data\Matteo\Embryo Illumination\Seg Illumination\run4_cellpose_NCandcaps\masks")
TMP_PATH.mkdir(parents=True, exist_ok=True)


def get_mask_path(experiment_name: str):
    """Generate mask file path from experiment name"""
    parts = experiment_name.split(".")
    name = "_".join(parts[1:])
    path = TMP_PATH / f"{name}_masks.npy"
    return path


# ============================================================================
# EXPERIMENT 1: BODY SEGMENTATION (Middle Plane)
# ============================================================================

class BodySegmentationMethod(CellposeSegmentationMethod):
    """
    Image at the body plane of the embryo.
    Segments the body and passes to pattern method
    Uses rescaling to work around CellposeSAM diameter issues
    """

    def __init__(self, model="cpsam", use_gpu=True, normlow=0, normhigh=5000, rescale_factor=0.06, **kwargs):
        super().__init__(model, use_gpu, normlow, normhigh, **kwargs)
        self.rescale_factor = rescale_factor
        logger.info(f"BodySegmentation initialized with rescale_factor={rescale_factor}")

    def segment(self, image):
        """Segment with rescaling to handle large embryo"""

        # Shrink for cellpose
        logger.debug(f"Original image shape: {image.shape}")
        image_rescaled = rescale(image, self.rescale_factor, preserve_range=True)
        logger.debug(f"Rescaled image shape: {image_rescaled.shape}")

        # Run parent Cellpose segmentation on small image
        masks_downscaled = super().segment(image_rescaled)

        # Unshrink to return full-size mask
        masks = resize(masks_downscaled, image.shape, order=0, preserve_range=True, anti_aliasing=False)
        masks = np.round(masks).astype(int)
        logger.debug(f"Upscaled mask shape: {masks.shape}")

        # If multiple objects found, keep only the largest
        if masks.max() > 1:
            props = measure.regionprops(masks)
            largest = max(props, key=lambda r: r.area)
            binary_mask = masks == largest.label
            logger.info(f"Cellpose found {masks.max()} objects, kept largest with area={largest.area}")
        else:
            binary_mask = masks > 0
            logger.info(f"Cellpose found 1 object")

        return binary_mask.astype(np.uint8)


class BodyCapsPatternMethod(PatternMethod):
    """
    Takes in cap parameters
    Creates caps mask using PCA method
    Saves the cap mask and returns all zeros on pattern generation
    """

    def __init__(self,
                 experiment_name,
                 camera_properties,
                 channel=None,
                 coverage=15,
                 **kwargs):

        super().__init__(experiment_name, camera_properties)
        self.channel = channel
        self.seg_channel_id = None
        self.coverage = coverage
        logger.info(f"BodyCapsPattern initialized with coverage={coverage}%")

    def initialize(self, experiment):
        super().initialize(experiment)
        channel = experiment.channels.get(self.channel, None)
        assert channel is not None, f"provided channel {self.channel} is not in experiment"

        self.seg_channel_id = channel.channel_id

        # Request segmentation data
        return [AcquiredImageRequest(self.seg_channel_id, needs_raw=False, needs_seg=True)]

    def make_caps(self, binary_mask):
        """
        Create end caps mask using PCA method

        Parameters:
        -----------
        binary_mask : np.ndarray
            Binary segmentation of whole embryo

        Returns:
        --------
        caps_mask : np.ndarray
            Binary mask with filled caps at both ends
        """
        coverage = self.coverage

        # Find embryo ends using PCA
        coords = np.argwhere(binary_mask)

        if len(coords) == 0:
            logger.warning("Empty embryo mask, returning zeros")
            return np.zeros_like(binary_mask, dtype=np.uint8)

        # PCA to find major axis
        center = coords.mean(axis=0)
        centered_coords = coords - center

        cov_matrix = np.cov(centered_coords.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        major_axis_idx = np.argmax(eigenvalues)
        major_axis = eigenvectors[:, major_axis_idx].real

        # Project onto major axis
        projections = np.dot(centered_coords, major_axis)
        embryo_length = projections.max() - projections.min()

        # Calculate illumination distance
        illuminate_distance = (coverage / 100) * embryo_length

        logger.info(f"Embryo length: {embryo_length:.1f}px, illuminating {illuminate_distance:.1f}px from each end")

        # Create cap masks
        end1_cap_mask = np.zeros_like(binary_mask, dtype=bool)
        end2_cap_mask = np.zeros_like(binary_mask, dtype=bool)

        min_projection = projections.min()
        max_projection = projections.max()

        for i, coord in enumerate(coords):
            projection = projections[i]
            dist_from_end1 = projection - min_projection
            dist_from_end2 = max_projection - projection

            y, x = coord

            if dist_from_end1 <= illuminate_distance and dist_from_end1 < dist_from_end2:
                end1_cap_mask[y, x] = True
            elif dist_from_end2 <= illuminate_distance:
                end2_cap_mask[y, x] = True

        # Fill with convex hull
        if end1_cap_mask.any():
            end1_cap_mask = morphology.convex_hull_image(end1_cap_mask)
            for y, x in np.argwhere(end1_cap_mask):
                centered = np.array([y, x]) - center
                proj = np.dot(centered, major_axis)
                if proj - min_projection > illuminate_distance:
                    end1_cap_mask[y, x] = False

        if end2_cap_mask.any():
            end2_cap_mask = morphology.convex_hull_image(end2_cap_mask)
            for y, x in np.argwhere(end2_cap_mask):
                centered = np.array([y, x]) - center
                proj = np.dot(centered, major_axis)
                if max_projection - proj > illuminate_distance:
                    end2_cap_mask[y, x] = False

        caps_mask = end1_cap_mask | end2_cap_mask

        logger.info(f"Created caps: {caps_mask.sum()} pixels")

        return caps_mask.astype(np.uint8)

    def generate(self, data: DataDock):
        """Generate caps mask and save to file"""

        # Get segmentation data
        binary_mask = data.data[self.seg_channel_id]["seg"].data

        # Create caps mask
        caps = self.make_caps(binary_mask)

        # Save to file for nc experiment to load
        caps_save_path = get_mask_path(self.experiment)
        np.save(caps_save_path, caps)
        logger.info(f"Saved caps mask to: {caps_save_path}")

        # Save debug visualization if enabled
        if self.save_debug:
            debug_path = TMP_PATH.parent / "Debug Output"
            debug_path.mkdir(exist_ok=True)

            # Save embryo mask
            mask_debug = debug_path / f"{self.experiment}_embryo_mask.tif"
            tifffile.imwrite(mask_debug, (binary_mask * 255).astype(np.uint8))

            # Save caps mask
            caps_debug = debug_path / f"{self.experiment}_caps_mask.tif"
            tifffile.imwrite(caps_debug, (caps * 255).astype(np.uint8))

            # Save overlay
            overlay = np.stack([binary_mask, caps, np.zeros_like(caps)], axis=-1)
            overlay_debug = debug_path / f"{self.experiment}_overlay.tif"
            tifffile.imwrite(overlay_debug, (overlay * 255).astype(np.uint8))

            logger.info(f"Saved debug visualizations to {debug_path}")

        # Return zeros (no actual illumination in this experiment)
        return np.zeros(binary_mask.shape, dtype=np.float32)


# ============================================================================
# EXPERIMENT 2: NC DETECTION AND ILLUMINATION (Bottom Plane)
# ============================================================================

class NucleusSegmentation(CellposeSegmentationMethod):
    """
    Images at the histone plane.
    Output the segmented nuclei
    Uses CellposeSAM
    """
    name = "segment_nuclei"


class NuclearCycleAppliesCap(PatternMethod):
    """
    Loads body cap from file if it exists yet
    Detects nuclear cycle, and determines whether to start running the pattern
    If not triggered, returns zeros
    If triggered, applies the caps mask for 45 minutes
    """

    def __init__(self,
                 experiment_name,
                 camera_properties,
                 channel=None,
                 pkl_file=None,
                 illumination_duration_minutes=45,
                 **kwargs):

        super().__init__(experiment_name, camera_properties)
        self.channel = channel
        self.seg_channel_id = None
        self.pkl_file = pkl_file
        self.illumination_duration_minutes = illumination_duration_minutes

        # Load Random Forest model
        with open(pkl_file, "rb") as f:
            self.forest_model: RandomForestClassifier = pickle.load(f)
        logger.info(f"Loaded Random Forest model from {pkl_file}")

        self.mask = None
        self.all_predictions = []

        # Timing tracking
        self.triggered = False
        self.trigger_time = None

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        super().initialize(experiment)

        channel = experiment.channels.get(self.channel, None)
        assert channel is not None, f"provided channel {self.channel} is not in experiment"

        self.seg_channel_id = channel.channel_id

        # Request segmentation only
        return [AcquiredImageRequest(self.seg_channel_id, needs_raw=False, needs_seg=True)]

    def predict_nc_cycle(self, nucleus_mask) -> str:
        """
        Predict nuclear cycle stage from segmented nuclei

        Parameters:
        -----------
        nucleus_mask : np.ndarray
            Labeled mask from Cellpose

        Returns:
        --------
        prediction : str
            Predicted NC stage
        """
        classifier = self.forest_model

        # Extract features matching your training data
        features = self._extract_nc_features(nucleus_mask, nucleus_mask)

        # Convert to feature vector in correct order
        feature_names = ['detected_diameter', 'nucleus_count', 'area_mean', 'area_std', 'area_cv',
                        'area_min', 'area_max', 'area_median', 'area_q25', 'area_q75',
                        'density', 'coverage', 'circularity_mean', 'circularity_std',
                        'eccentricity_mean', 'eccentricity_std', 'solidity_mean', 'solidity_std',
                        'aspect_ratio_mean', 'aspect_ratio_std', 'aspect_ratio_max',
                        'elongation_mean', 'elongation_std', 'pct_elongated',
                        'intensity_mean', 'intensity_std', 'intensity_cv',
                        'nn_dist_mean', 'nn_dist_std', 'nn_dist_min', 'nn_dist_median',
                        'clustering_index']

        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        prediction = classifier.predict(feature_vector)[0]

        logger.info(f"Predicted NC stage: {prediction}")

        return prediction

    def _extract_nc_features(self, image, masks):
        """Extract 32 features for NC classification"""
        features = {}

        props = measure.regionprops(masks, intensity_image=image)

        # Estimate diameter
        if len(props) > 0:
            diameters = [np.sqrt(p.area / np.pi) * 2 for p in props]
            detected_diameter = np.median(diameters)
        else:
            detected_diameter = 30

        features['detected_diameter'] = detected_diameter

        if len(props) == 0:
            return self._get_empty_nc_features(detected_diameter)

        # Nucleus count
        features['nucleus_count'] = len(props)

        # Area features
        areas = [p.area for p in props]
        features['area_mean'] = np.mean(areas)
        features['area_std'] = np.std(areas)
        features['area_cv'] = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        features['area_min'] = np.min(areas)
        features['area_max'] = np.max(areas)
        features['area_median'] = np.median(areas)
        features['area_q25'] = np.percentile(areas, 25)
        features['area_q75'] = np.percentile(areas, 75)

        # Density
        image_area = image.shape[0] * image.shape[1]
        features['density'] = len(props) / image_area
        features['coverage'] = np.sum(areas) / image_area

        # Shape features
        circularities = [p.area / (np.pi * (p.major_axis_length/2)**2) if p.major_axis_length > 0 else 0
                         for p in props]
        eccentricities = [p.eccentricity for p in props]
        solidities = [p.solidity for p in props]
        aspect_ratios = [p.major_axis_length / p.minor_axis_length if p.minor_axis_length > 0 else 1
                         for p in props]
        elongations = [1 - (p.minor_axis_length / p.major_axis_length) if p.major_axis_length > 0 else 0
                       for p in props]

        features['circularity_mean'] = np.mean(circularities)
        features['circularity_std'] = np.std(circularities)
        features['eccentricity_mean'] = np.mean(eccentricities)
        features['eccentricity_std'] = np.std(eccentricities)
        features['solidity_mean'] = np.mean(solidities)
        features['solidity_std'] = np.std(solidities)
        features['aspect_ratio_mean'] = np.mean(aspect_ratios)
        features['aspect_ratio_std'] = np.std(aspect_ratios)
        features['aspect_ratio_max'] = np.max(aspect_ratios)
        features['elongation_mean'] = np.mean(elongations)
        features['elongation_std'] = np.std(elongations)

        elongated_count = sum(1 for ar in aspect_ratios if ar > 1.5)
        features['pct_elongated'] = elongated_count / len(props) * 100

        # Intensity features
        intensities = [p.mean_intensity for p in props]
        features['intensity_mean'] = np.mean(intensities)
        features['intensity_std'] = np.std(intensities)
        features['intensity_cv'] = np.std(intensities) / np.mean(intensities) if np.mean(intensities) > 0 else 0

        # Spatial features
        centroids = np.array([p.centroid for p in props])
        if len(centroids) > 1:
            dist_matrix = spatial.distance.cdist(centroids, centroids)
            np.fill_diagonal(dist_matrix, np.inf)
            nearest_dists = np.min(dist_matrix, axis=1)

            features['nn_dist_mean'] = np.mean(nearest_dists)
            features['nn_dist_std'] = np.std(nearest_dists)
            features['nn_dist_min'] = np.min(nearest_dists)
            features['nn_dist_median'] = np.median(nearest_dists)

            mean_all_dist = np.mean(dist_matrix[dist_matrix != np.inf])
            features['clustering_index'] = features['nn_dist_mean'] / mean_all_dist if mean_all_dist > 0 else 1
        else:
            features['nn_dist_mean'] = 0
            features['nn_dist_std'] = 0
            features['nn_dist_min'] = 0
            features['nn_dist_median'] = 0
            features['clustering_index'] = 1

        return features

    def _get_empty_nc_features(self, detected_diameter=0):
        """Return zero features when no nuclei detected"""
        return {
            'detected_diameter': detected_diameter, 'nucleus_count': 0,
            'area_mean': 0, 'area_std': 0, 'area_cv': 0,
            'area_min': 0, 'area_max': 0, 'area_median': 0,
            'area_q25': 0, 'area_q75': 0,
            'density': 0, 'coverage': 0,
            'circularity_mean': 0, 'circularity_std': 0,
            'eccentricity_mean': 0, 'eccentricity_std': 0,
            'solidity_mean': 0, 'solidity_std': 0,
            'aspect_ratio_mean': 0, 'aspect_ratio_std': 0, 'aspect_ratio_max': 0,
            'elongation_mean': 0, 'elongation_std': 0, 'pct_elongated': 0,
            'intensity_mean': 0, 'intensity_std': 0, 'intensity_cv': 0,
            'nn_dist_mean': 0, 'nn_dist_std': 0, 'nn_dist_min': 0, 'nn_dist_median': 0,
            'clustering_index': 0
        }

    def choose_to_apply(self) -> bool:
        """
        Check if last 2 predictions are both NC10-13
        Returns True if pattern should be applied
        """
        predicted_ncs = self.all_predictions

        # Need at least 2 predictions
        if len(predicted_ncs) < 2:
            logger.debug(f"Only {len(predicted_ncs)} prediction(s), need 2 for trigger")
            return False

        # Get last 2
        last_two = predicted_ncs[-2:]

        # Define trigger range (NC10-13 including M-phases)
        trigger_stages = ['NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M']

        # Check if both in range
        both_in_range = all(nc in trigger_stages for nc in last_two)

        if both_in_range and not self.triggered:
            logger.info(f"🔔 TRIGGER! Last 2 predictions: {last_two[0]}, {last_two[1]}")
            self.triggered = True

        return both_in_range

    def check_for_mask(self):
        """Try to load caps mask from file"""
        mask_path = get_mask_path(self.experiment)

        if not mask_path.exists():
            logger.debug(f"Caps mask not yet available at {mask_path}")
            return False

        self.mask = np.load(mask_path).astype(np.float16)
        logger.info(f"Loaded caps mask from {mask_path}: {self.mask.sum()} pixels")

        return True

    def check_illumination_duration(self, current_time):
        """Check if we should stop illumination after 45 minutes"""
        if self.trigger_time is None:
            return True

        elapsed_minutes = (current_time - self.trigger_time) / 60
        still_illuminating = elapsed_minutes < self.illumination_duration_minutes

        if not still_illuminating:
            logger.info(f"Illumination complete: {elapsed_minutes:.1f} min elapsed")

        return still_illuminating

    def generate(self, data_dock: DataDock) -> np.ndarray:
        """
        Classify NC stage and apply pattern if triggered
        """
        current_time = data_dock.time_seconds

        # Get nucleus segmentation
        nucleus_mask = data_dock.data[self.seg_channel_id]["seg"].data

        # Predict NC stage
        predicted_nuclear_cycle = self.predict_nc_cycle(nucleus_mask)
        self.all_predictions.append(predicted_nuclear_cycle)

        logger.info(f"Frame {len(self.all_predictions)}: Classified as {predicted_nuclear_cycle}")

        # Check if we should apply pattern
        should_apply = self.choose_to_apply()

        # Try to load caps mask if we don't have it yet
        if not self.mask:
            self.check_for_mask()

        # Apply pattern if triggered and within duration
        if should_apply:
            # Record trigger time if first time
            if self.trigger_time is None:
                self.trigger_time = current_time
                logger.info(f"Starting illumination at t={current_time}s")

            # Check if still within 45 minute window
            if self.check_illumination_duration(current_time):
                if self.mask is not None:
                    elapsed_min = (current_time - self.trigger_time) / 60
                    logger.info(f"Applying caps pattern ({elapsed_min:.1f}/{self.illumination_duration_minutes} min)")
                    return self.mask
                else:
                    logger.warning("Triggered but caps mask not available yet")
                    return np.zeros(nucleus_mask.shape, dtype=np.float32)

        # Not triggered or illumination finished
        return np.zeros(nucleus_mask.shape, dtype=np.float32)


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run PyCLM dual-experiment"""

    working_directory = Path(r"Z:\Microscopy Data\Matteo\Embryo Illumination\Seg Illumination\run4_cellpose_NCandcaps")

    segmentation_methods = {
        "body_segmentation": BodySegmentationMethod,
        "nucleus_segmentation": NucleusSegmentation,
    }

    pattern_methods = {
        "make_caps": BodyCapsPatternMethod,
        "apply_caps": NuclearCycleAppliesCap,
    }

    print("=" * 70)
    print("DUAL EXPERIMENT: BODY + NC-TRIGGERED ILLUMINATION")
    print("=" * 70)
    print("Experiment 1 (body.toml):")
    print("  - Runs ONCE at start")
    print("  - Images middle body plane")
    print("  - Rescales, segments, creates caps")
    print("  - Saves caps mask to file")
    print("")
    print("Experiment 2 (nc.toml):")
    print("  - Runs every 2 minutes")
    print("  - Images bottom nuclei plane")
    print("  - Classifies NC stage with ML")
    print("  - Applies caps when NC10-13 × 2")
    print("  - Illuminates for 45 minutes")
    print("=" * 70)
    print(f"Working directory: {working_directory}")
    print(f"Masks: {TMP_PATH}")
    print("=" * 70)

    run_pyclm(
        working_directory,
        segmentation_methods=segmentation_methods,
        pattern_methods=pattern_methods
    )


if __name__ == "__main__":
    main()