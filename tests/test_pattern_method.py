"""PatternMethod base-class behaviour: binning bookkeeping and PatternReview construction."""

import h5py
import numpy as np
import pytest
from helpers import make_experiment

from pyclm.core.patterns import ROI, CameraProperties
from pyclm.core.patterns.pattern import PatternReview
from pyclm.core.patterns.static_patterns import FullOnPattern


def test_update_binning_keeps_integer_shape():
    method = FullOnPattern()
    exp = make_experiment("exp.00", stim_binning=4)

    method.configure_system("exp.00", CameraProperties(ROI(0, 0, 800, 600), 0.33), exp)

    assert method.pattern_shape == (150, 200)
    assert all(isinstance(v, int) for v in method.pattern_shape)
    assert method.pixel_size_um == pytest.approx(1.32)

    method.update_binning(2)

    assert method.pattern_shape == (300, 400)
    assert all(isinstance(v, int) for v in method.pattern_shape)
    assert method.pixel_size_um == pytest.approx(0.66)
    assert method.generate(None).shape == (300, 400)


def test_pattern_review_constructs_from_toml_kwargs(tmp_path):
    h5_path = tmp_path / "previous.hdf5"
    dmd = np.full((4, 4), 7, dtype=np.uint8)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("00000/channel_545/data", data=np.zeros((2, 2), np.uint16))
        f.create_dataset("00000/stim_aq/dmd", data=dmd)
        f.create_dataset("00001/channel_638/data", data=np.zeros((2, 2), np.uint16))

    # constructed the way PatternProcess.request_method does: kwargs only
    method = PatternReview(h5fp=str(h5_path), channel="545")

    first = method.generate(None)
    assert np.array_equal(first, dmd)
    # no further timepoint contains channel_545 → empty array
    assert method.generate(None).size == 0
