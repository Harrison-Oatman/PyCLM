"""PatternMethod base-class behaviour: binning bookkeeping."""

import numpy as np
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


class _At:
    def __init__(self, seconds):
        self.time = seconds


def _bar_centroid_y(method, seconds):
    pattern = method.generate(_At(seconds))
    ys = np.nonzero(pattern)[0]
    return ys.mean()


@pytest.mark.parametrize("cls", ["bar", "sawtooth", "bar_bounce"])
def test_negative_bar_speed_reverses_the_direction(cls):
    from pyclm.core.patterns import known_models

    def make(speed):
        m = known_models[cls](duty_cycle=0.1, bar_speed=speed, period=100)
        m.configure_system(
            "exp.00",
            CameraProperties(ROI(0, 0, 10, 100), 1.0),
            make_experiment("exp.00"),
        )
        return m

    # one bar per 100 um over 100 px at 1 um/px: a single 10-px bar, so the
    # centroid of the lit rows says where it is
    forward, backward = make(10.0), make(-10.0)
    assert forward.period_time == backward.period_time == 10.0

    def travel(m):
        # displacement of the bar between 2.5 and 4.5 min: at 10 um/min the
        # bar sits well inside the 100 um frame at both times, either way
        return _bar_centroid_y(m, 270) - _bar_centroid_y(m, 150)

    d_forward, d_backward = travel(forward), travel(backward)
    assert d_forward == pytest.approx(20, abs=1.5)  # 10 um/min for 2 min, towards +y
    assert d_backward == pytest.approx(-20, abs=1.5)  # the same distance towards -y
    # the lit fraction is the duty cycle either way (half of it for the
    # sawtooth, whose bar is a ramp)
    expected = 0.05 if cls == "sawtooth" else 0.1
    assert forward.generate(_At(0)).mean() == pytest.approx(expected, abs=0.02)
    assert backward.generate(_At(0)).mean() == pytest.approx(expected, abs=0.02)


def test_bar_period_follows_a_speed_change():
    from pyclm.core.patterns import known_models

    m = known_models["bar"](duty_cycle=0.2, bar_speed=1.0, period=100)
    assert m.period_time == 100.0
    applied, _refused = m.update(bar_speed=4.0)
    assert applied == {"bar_speed": 4.0}
    assert m.period_time == 25.0
    with pytest.raises(ValueError, match="non-zero"):
        known_models["sawtooth"](bar_speed=0)
