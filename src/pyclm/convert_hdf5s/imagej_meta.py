import numpy as np


def grayscale_lut() -> np.ndarray:
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp, ramp, ramp])


def cyan_lut() -> np.ndarray:
    zero = np.zeros(256, dtype=np.uint8)
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([zero, ramp, ramp])


def yellow_lut() -> np.ndarray:
    zero = np.zeros(256, dtype=np.uint8)
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp, ramp, zero])


def build_metadata(has_seg: bool) -> dict:
    """
    ImageJ metadata for a tcyx stack of [data, (seg,) pattern] channels.
    """
    luts = [grayscale_lut()]
    ranges = [0, 5000]

    if has_seg:
        luts.append(yellow_lut())
        ranges += [0, 1]

    luts.append(cyan_lut())
    ranges += [0, 1000]

    return {
        "axes": "tcyx",
        "mode": "composite",
        "LUTs": luts,
        "Ranges": ranges,
    }
