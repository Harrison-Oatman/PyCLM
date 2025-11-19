"""
Dual Experiment: Body Segmentation + NC-Triggered Illumination
Following the working run3 architecture - all-in-one pattern modules
No inheritance from CellposeSegmentationMethod (avoids crashes)
"""

from pyclm import run_pyclm
from pyclm.core.patterns.feedback_control_patterns import PerCellPatternMethod, AcquiredImageRequest, DataDock

import numpy as np
from cellpose import models
from skimage import morphology, measure
from skimage.transform import rescale, resize
from scipy import ndimage, spatial
from pathlib import Path
import tifffile
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# Path where caps masks will be saved (shared between experiments)
TMP_PATH = Path(r"Z:\Microscopy Data\Matteo\Embryo Illumination\Seg Illumination\run4_cellpose_NCandcaps\masks")
TMP_PATH.mkdir(parents=True, exist_ok=True)

# Shared model instances to avoid loading multiple copies
_BODY_MODEL = None
_NC_MODEL = None

def get_body_model(gpu=True, model_type='cyto3'):
    """Get or create shared body Cellpose model"""
    global _BODY_MODEL
    if _BODY_MODEL is None:
        logger.info(f"Creating shared body Cellpose model: {model_type}")
        _BODY_MODEL = models.CellposeModel(gpu=gpu, model_type=model_type)
    return _BODY_MODEL

def get_nc_model(gpu=True, model_type='nuclei'):
    """Get or create shared NC Cellpose model"""
    global _NC_MODEL
    if _NC_MODEL is None:
        logger.info(f"Creating shared NC Cellpose model: {model_type}")
        _NC_MODEL = models.CellposeModel(gpu=gpu, model_type=model_type)
    return _NC_MODEL


def get_mask_path(experiment_name: str):
    """Generate mask file path from experiment name"""
    parts = experiment_name.split(".")
    name = "_".join(parts[1:])
    path = TMP_PATH / f"{name}_masks.npy"
    return path


# ============================================================================
# EXPERIMENT 1: BODY CAPS CREATION (Middle Plane)
# ============================================================================

class BodyCapsPattern(PerCellPatternMethod):
    """
    All-in-one for body experiment:
    - Segments whole embryo with Cellpose (using rescaling)
    - Creates caps mask with PCA
    - Saves to file
    - Returns zeros (no illumination)
    """

    name = "make_caps"

    def __init__(self,
                 experiment_name,
                 camera_properties,
                 channel=None,
                 coverage=15,
                 cellpose_model='cyto3',
                 gpu=True,
                 rescale_factor=0.06,
                 normlow=0,
                 normhigh=15000,
                 save_debug=True,
                 **kwargs):

        super().__init__(experiment_name, camera_properties, channel, **kwargs)

        self.coverage = coverage
        self.cellpose_model = cellpose_model
        self.gpu = gpu
        self.rescale_factor = rescale_factor
        self.normlow = normlow
        self.normhigh = normhigh
        self.save_debug = save_debug

        # Initialize Cellpose model
        logger.info(f"Initializing Cellpose model for body: {cellpose_model}")
        self.model = get_body_model(gpu=gpu, model_type=cellpose_model)

        logger.info(f"BodyCapsPattern initialized: coverage={coverage}%, rescale={rescale_factor}")

    def initialize(self, experiment):
        super().initialize(experiment)

        channel = experiment.channels.get(self.channel, None)
        assert channel is not None, f"provided channel {self.channel} is not in experiment"

        self.seg_channel_id = channel.channel_id

        # Request raw image
        return [AcquiredImageRequest(channel.channel_id, needs_raw=True, needs_seg=False)]

    def segment_embryo(self, image):
        """Segment whole embryo with rescaling"""
        try:
            # Normalize
            image_norm = (image.astype(np.float32) - self.normlow) / (self.normhigh - self.normlow)
            image_norm = np.clip(image_norm, 0, 1)

            # Rescale down
            logger.debug(f"Original shape: {image.shape}")
            image_small = rescale(image_norm, self.rescale_factor, preserve_range=True)
            logger.debug(f"Rescaled shape: {image_small.shape}")

            # Run Cellpose on small image
            masks_small, flows, styles = self.model.eval(
                image_small,
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )

            # Resize back to original size
            masks = resize(masks_small, image.shape, order=0, preserve_range=True, anti_aliasing=False)
            masks = np.round(masks).astype(int)

            # Keep only largest object
            if masks.max() == 0:
                logger.warning("No embryo found")
                return np.zeros_like(image, dtype=np.uint8)

            if masks.max() > 1:
                props = measure.regionprops(masks)
                largest = max(props, key=lambda r: r.area)
                binary_mask = masks == largest.label
                logger.info(f"Found {masks.max()} objects, kept largest: {largest.area}px")
            else:
                binary_mask = masks > 0
                logger.info(f"Found 1 embryo")

            return binary_mask.astype(np.uint8)

        except Exception as e:
            logger.error(f"Error segmenting embryo: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros_like(image, dtype=np.uint8)

    def make_caps(self, binary_mask):
        """Create end caps using PCA"""
        coords = np.argwhere(binary_mask)

        if len(coords) == 0:
            return np.zeros_like(binary_mask, dtype=np.uint8)

        # PCA
        center = coords.mean(axis=0)
        centered_coords = coords - center
        cov_matrix = np.cov(centered_coords.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        major_axis_idx = np.argmax(eigenvalues)
        major_axis = eigenvectors[:, major_axis_idx].real

        # Project and measure
        projections = np.dot(centered_coords, major_axis)
        embryo_length = projections.max() - projections.min()
        illuminate_distance = (self.coverage / 100) * embryo_length

        # Create caps
        end1_cap_mask = np.zeros_like(binary_mask, dtype=bool)
        end2_cap_mask = np.zeros_like(binary_mask, dtype=bool)

        min_proj = projections.min()
        max_proj = projections.max()

        for i, coord in enumerate(coords):
            proj = projections[i]
            dist1 = proj - min_proj
            dist2 = max_proj - proj
            y, x = coord

            if dist1 <= illuminate_distance and dist1 < dist2:
                end1_cap_mask[y, x] = True
            elif dist2 <= illuminate_distance:
                end2_cap_mask[y, x] = True

        # Convex hull fill
        if end1_cap_mask.any():
            end1_cap_mask = morphology.convex_hull_image(end1_cap_mask)
            for y, x in np.argwhere(end1_cap_mask):
                if np.dot(np.array([y, x]) - center, major_axis) - min_proj > illuminate_distance:
                    end1_cap_mask[y, x] = False

        if end2_cap_mask.any():
            end2_cap_mask = morphology.convex_hull_image(end2_cap_mask)
            for y, x in np.argwhere(end2_cap_mask):
                if max_proj - np.dot(np.array([y, x]) - center, major_axis) > illuminate_distance:
                    end2_cap_mask[y, x] = False

        caps_mask = end1_cap_mask | end2_cap_mask
        logger.info(f"Created caps: {caps_mask.sum()}px, length={embryo_length:.1f}px")

        return caps_mask.astype(np.uint8)

    def generate(self, data_dock: DataDock):
        """Segment, create caps, save to file"""
        try:
            raw_image = data_dock.data[self.seg_channel_id]["raw"].data

            # Segment embryo
            binary_mask = self.segment_embryo(raw_image)

            # Create caps
            caps = self.make_caps(binary_mask)

            # Save to file
            caps_save_path = get_mask_path(self.experiment)
            np.save(caps_save_path, caps)
            logger.info(f"Saved caps mask to: {caps_save_path}")

            # Save debug TIFFs
            if self.save_debug:
                try:
                    # Try network path first
                    debug_path = TMP_PATH.parent / "Debug Output"
                    debug_path.mkdir(exist_ok=True)

                    tifffile.imwrite(debug_path / f"{self.experiment}_embryo_mask.tif",
                                   (binary_mask * 255).astype(np.uint8))
                    tifffile.imwrite(debug_path / f"{self.experiment}_caps_mask.tif",
                                   (caps * 255).astype(np.uint8))

                    overlay = np.stack([binary_mask, caps, np.zeros_like(caps)], axis=-1)
                    tifffile.imwrite(debug_path / f"{self.experiment}_overlay.tif",
                                   (overlay * 255).astype(np.uint8))

                    logger.info(f"Saved debug TIFFs to {debug_path}")
                except Exception as e:
                    # Fall back to local path if network fails
                    logger.warning(f"Network debug path failed: {e}, using local path")
                    try:
                        debug_path = Path(r"C:\Temp\Embryo_Debug")
                        debug_path.mkdir(parents=True, exist_ok=True)

                        tifffile.imwrite(debug_path / f"{self.experiment}_embryo_mask.tif",
                                       (binary_mask * 255).astype(np.uint8))
                        tifffile.imwrite(debug_path / f"{self.experiment}_caps_mask.tif",
                                       (caps * 255).astype(np.uint8))

                        overlay = np.stack([binary_mask, caps, np.zeros_like(caps)], axis=-1)
                        tifffile.imwrite(debug_path / f"{self.experiment}_overlay.tif",
                                       (overlay * 255).astype(np.uint8))

                        logger.info(f"Saved debug TIFFs to local path: {debug_path}")
                    except Exception as e2:
                        logger.error(f"Failed to save debug TIFFs even to local path: {e2}")
                        # Continue anyway - debug output is not critical

            return np.zeros(binary_mask.shape, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error in body caps generation: {e}")
            import traceback
            traceback.print_exc()
            h, w = self.pattern_shape
            return np.zeros((int(h), int(w)), dtype=np.float32)


# ============================================================================
# EXPERIMENT 2: NC DETECTION AND CAPS APPLICATION (Bottom Plane)
# ============================================================================

class NCTriggeredCapsPattern(PerCellPatternMethod):
    """
    NC detection and caps application
    """

    name = "apply_caps"

    def __init__(self,
                 experiment_name,
                 camera_properties,
                 channel=None,
                 pkl_file=None,
                 cellpose_model='nuclei',
                 gpu=True,
                 normlow=0,
                 normhigh=65535,
                 illumination_duration_minutes=60,
                 **kwargs):

        super().__init__(experiment_name, camera_properties, channel, **kwargs)

        self.cellpose_model = cellpose_model
        self.gpu = gpu
        self.normlow = normlow
        self.normhigh = normhigh
        self.illumination_duration_minutes = illumination_duration_minutes

        # Initialize Cellpose
        logger.info(f"Initializing Cellpose for nuclei: {cellpose_model}")
        self.model = get_nc_model(gpu=gpu, model_type=cellpose_model)

        # Load RF model - try multiple methods
        logger.info(f"Loading RF model from {pkl_file}")

        # Method 1: Try joblib (sklearn's recommended way)
        try:
            import joblib
            loaded_obj = joblib.load(pkl_file)
            # Check if it's a dict containing the model
            if isinstance(loaded_obj, dict):
                logger.info(f"Loaded dict with keys: {list(loaded_obj.keys())}")
                # Try common keys
                if 'model' in loaded_obj:
                    self.forest_model = loaded_obj['model']
                    logger.info(f"[OK] Found model in dict key: 'model'")
                elif 'classifier' in loaded_obj:
                    self.forest_model = loaded_obj['classifier']
                    logger.info(f"[OK] Found model in dict key: 'classifier'")
                elif 'forest' in loaded_obj:
                    self.forest_model = loaded_obj['forest']
                    logger.info(f"[OK] Found model in dict key: 'forest'")
                else:
                    # Take the first object with a predict method
                    for key, val in loaded_obj.items():
                        if hasattr(val, 'predict'):
                            self.forest_model = val
                            logger.info(f"[OK] Found model in dict key: '{key}'")
                            break
                    else:
                        raise RuntimeError(f"No model with predict() found in dict keys: {list(loaded_obj.keys())}")
            else:
                self.forest_model = loaded_obj
                logger.info(f"[OK] RF model loaded with joblib")
        except Exception as e1:
            logger.warning(f"joblib load failed: {e1}, trying pickle...")

            # Method 2: Try standard pickle with warnings suppressed
            try:
                import warnings
                with open(pkl_file, 'rb') as f:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        loaded_obj = pickle.load(f)

                # Same dict handling as above
                if isinstance(loaded_obj, dict):
                    logger.info(f"Loaded dict with keys: {list(loaded_obj.keys())}")
                    if 'model' in loaded_obj:
                        self.forest_model = loaded_obj['model']
                        logger.info(f"[OK] Found model in dict key: 'model'")
                    elif 'classifier' in loaded_obj:
                        self.forest_model = loaded_obj['classifier']
                        logger.info(f"[OK] Found model in dict key: 'classifier'")
                    elif 'forest' in loaded_obj:
                        self.forest_model = loaded_obj['forest']
                        logger.info(f"[OK] Found model in dict key: 'forest'")
                    else:
                        for key, val in loaded_obj.items():
                            if hasattr(val, 'predict'):
                                self.forest_model = val
                                logger.info(f"[OK] Found model in dict key: '{key}'")
                                break
                        else:
                            raise RuntimeError(f"No model with predict() found in dict keys: {list(loaded_obj.keys())}")
                else:
                    self.forest_model = loaded_obj

                logger.info(f"[OK] RF model loaded with pickle")
            except Exception as e2:
                logger.error(f"Both loading methods failed!")
                logger.error(f"Error 1 (joblib): {e1}")
                logger.error(f"Error 2 (pickle): {e2}")
                raise RuntimeError(f"Cannot load RF model")

        self.mask = None
        self.all_predictions = []
        self.triggered = False
        self.trigger_time = None

    def initialize(self, experiment):
        super().initialize(experiment)

        channel = experiment.channels.get(self.channel, None)
        assert channel is not None, f"channel {self.channel} not found"

        self.seg_channel_id = channel.channel_id

        return [AcquiredImageRequest(channel.channel_id, needs_raw=True, needs_seg=False)]

    def segment_nuclei(self, image):
        """Segment nuclei"""
        try:
            image_norm = (image.astype(np.float32) - self.normlow) / (self.normhigh - self.normlow)
            image_norm = np.clip(image_norm, 0, 1)

            masks, flows, styles = self.model.eval(image_norm, flow_threshold=0.4, cellprob_threshold=0.0)

            return masks
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            return np.zeros_like(image, dtype=int)

    def extract_features(self, image, masks):
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
            return self._get_empty_features(detected_diameter)

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

    def _get_empty_features(self, detected_diameter=0):
        """Return zero features when no nuclei"""
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

    def predict_nc_cycle(self, features):
        """Classify NC stage"""
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

        prediction = self.forest_model.predict(feature_vector)[0]
        return prediction

    def check_trigger(self):
        """Check if last 2 predictions are NC10-13"""
        if len(self.all_predictions) < 2:
            return False

        last_two = self.all_predictions[-2:]
        trigger_stages = ['NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M']

        both_in_range = all(nc in trigger_stages for nc in last_two)

        if both_in_range and not self.triggered:
            logger.info(f"*** TRIGGER! {last_two[0]}, {last_two[1]} ***")
            self.triggered = True

        return both_in_range

    def check_for_mask(self):
        """Load caps from file"""
        if self.mask is not None:
            return True

        mask_path = get_mask_path(self.experiment)
        if not mask_path.exists():
            return False

        self.mask = np.load(mask_path).astype(np.float32)
        logger.info(f"Loaded caps: {mask_path}, {self.mask.sum()}px")
        return True

    def check_duration(self, current_time):
        """Check if within 60 min window"""
        if self.trigger_time is None:
            return True

        elapsed_min = (current_time - self.trigger_time) / 60
        return elapsed_min < self.illumination_duration_minutes

    def generate(self, data_dock: DataDock):
        """Main logic: classify, trigger, apply"""
        try:
            current_time = data_dock.time_seconds
            raw_image = data_dock.data[self.seg_channel_id]["raw"].data

            # Segment nuclei
            masks = self.segment_nuclei(raw_image)

            # Extract features and classify
            features = self.extract_features(raw_image, masks)
            nc_stage = self.predict_nc_cycle(features)
            self.all_predictions.append(nc_stage)

            logger.info(f"Frame {len(self.all_predictions)}: {nc_stage}")

            # Check trigger
            should_apply = self.check_trigger()

            # Try loading mask
            self.check_for_mask()

            # Apply if triggered
            if should_apply:
                if self.trigger_time is None:
                    self.trigger_time = current_time
                    logger.info(f"Illumination START at t={current_time}s")

                if self.check_duration(current_time):
                    if self.mask is not None:
                        elapsed = (current_time - self.trigger_time) / 60
                        logger.info(f"ILLUMINATING ({elapsed:.1f}/{self.illumination_duration_minutes}min)")
                        return self.mask
                    else:
                        logger.warning("Triggered but no mask yet")
                        return np.zeros_like(raw_image, dtype=np.float32)
                else:
                    logger.info("Illumination COMPLETE")

            return np.zeros_like(raw_image, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            h, w = self.pattern_shape
            return np.zeros((int(h), int(w)), dtype=np.float32)


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run dual experiment"""

    working_directory = Path(r"Z:\Microscopy Data\Matteo\Embryo Illumination\Seg Illumination\run4_cellpose_NCandcaps")

    pattern_methods = {
        "make_caps": BodyCapsPattern,
        "apply_caps": NCTriggeredCapsPattern,
    }

    print("="*70)
    print("DUAL EXPERIMENT: BODY + NC-TRIGGERED ILLUMINATION")
    print("="*70)
    print("Exp1 (body.toml): Segment embryo → Create caps → Save")
    print("Exp2 (nc.toml): Segment nuclei → Classify → Apply caps if NC10-13×2")
    print("="*70)
    print(f"Working dir: {working_directory}")
    print(f"Masks: {TMP_PATH}")
    print("="*70)

    run_pyclm(working_directory, pattern_methods=pattern_methods)


if __name__ == "__main__":
    main()