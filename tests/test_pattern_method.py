"""PatternMethod base-class behaviour: binning bookkeeping."""

import pytest
from helpers import make_experiment

from pyclm.core.patterns import ROI, CameraProperties
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
