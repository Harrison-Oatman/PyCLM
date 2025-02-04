import re
import numpy as np


def get_mapping(projector_api):

    mapping = projector_api.load_mapping(projector_api.get_projection_device())

    m = mapping.get_map().values()

    # horrible regex, TODO use ast instead
    pattern = r"AffineTransform\[\[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\], \[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]\]"

    # Find all matches
    matches = re.findall(pattern, m.to_string())

    M = np.stack(np.array(matches).astype(float)).reshape(-1, 2, 3)
    at = np.median(M, axis=0)

    print(f"Affine Transform from calibration: {at}")
    return at