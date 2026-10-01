"""The open-loop pulse trains: global_pulse, pulse_schedule, blinking_bar."""

import numpy as np
import pytest
from helpers import make_experiment

from pyclm.core.patterns import ROI, CameraProperties, known_models


class _At:
    """The only thing these methods read off a context: the elapsed time."""

    def __init__(self, minutes):
        self.time = minutes * 60


def make(name, **kwargs):
    method = known_models[name](**kwargs)
    method.configure_system(
        "exp.00",
        # ROI(x_offset, y_offset, width, height): a (100, 10) pattern,
        # 100 um along the bar axis at 1 um/px
        CameraProperties(ROI(0, 0, 10, 100), 1.0),
        make_experiment("exp.00"),
    )
    return method


def lit_fraction(method, minutes):
    return method.generate(_At(minutes)).mean()


# --------------------------------------------------------------- global_pulse


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [
        (0, 1.0),  # first on-edge at start_min + phase_min = 0
        (19.9, 1.0),
        (20, 0.0),  # on_min elapsed
        (99, 0.0),
        (100, 1.0),  # next period
        (119.9, 1.0),
        (120, 0.0),
    ],
)
def test_global_pulse_is_uniform_and_periodic(minutes, expected):
    # position 3: 20 min on out of every 100
    method = make("global_pulse", period_min=100, on_min=20)
    pattern = method.generate(_At(minutes))

    assert pattern.shape == (100, 10)
    assert set(np.unique(pattern)) == {expected}


def test_global_pulse_phase_sets_the_first_on_edge():
    # a bar at 1 um/min crosses a 100 um field's centre at 50 min
    method = make("global_pulse", period_min=400, on_min=80, phase_min=50)

    assert not method.is_on(49.9)
    assert method.is_on(50)
    assert method.is_on(129.9)
    assert not method.is_on(130)
    assert method.is_on(450)  # one period later


def test_global_pulse_respects_the_stimulation_window():
    method = make("global_pulse", period_min=100, on_min=20, start_min=30, stop_min=535)

    assert not method.is_on(29.9)  # nothing before the window opens
    assert method.is_on(30)  # the window's first on-edge
    assert not method.is_on(50)
    assert method.is_on(130)
    assert method.is_on(534.9)  # the window closes mid-pulse
    assert not method.is_on(535)
    assert not method.is_on(630)


def test_global_pulse_rejects_impossible_gates():
    with pytest.raises(ValueError, match="period_min must be positive"):
        make("global_pulse", period_min=0, on_min=0)
    with pytest.raises(ValueError, match="on_min must be between"):
        make("global_pulse", period_min=100, on_min=120)


# ------------------------------------------------------------- pulse_schedule


def test_pulse_schedule_lights_the_listed_intervals():
    # position 12: 20 min at 1 h and at 9 h
    method = make("pulse_schedule", pulses=[[60, 20], [540, 20]])

    for off in (0, 59.9, 80, 300, 539.9, 560, 1200):
        assert lit_fraction(method, off) == 0.0
    for on in (60, 79.9, 540, 559.9):
        assert lit_fraction(method, on) == 1.0


def test_pulse_schedule_is_measured_from_the_start_of_stimulation():
    method = make("pulse_schedule", pulses=[[60, 20]], start_min=30)

    assert not method.is_on(60)
    assert method.is_on(90)
    assert not method.is_on(110)


def test_pulse_schedule_rejects_malformed_pulses():
    with pytest.raises(ValueError, match="pair"):
        make("pulse_schedule", pulses=[[60, 20, 5]])
    with pytest.raises(ValueError, match="duration must be positive"):
        make("pulse_schedule", pulses=[[60, 0]])


# --------------------------------------------------------------- blinking_bar


def test_blinking_bar_is_the_stationary_bar_gated_in_time():
    # position 14: 100 um stripes at 20% duty, 20 min on out of every 100
    method = make(
        "blinking_bar", spacing_um=100, duty_cycle=0.20, period_min=100, on_min=20
    )
    stationary = make("bar", bar_speed=0, duty_cycle=0.20, period=100)

    on = method.generate(_At(10))
    np.testing.assert_array_equal(on, stationary.generate(_At(10)))
    assert on.mean() == pytest.approx(0.20, abs=0.01)

    off = method.generate(_At(30))
    assert off.shape == on.shape
    assert not off.any()


def test_blinking_bar_offset_moves_the_stripe_edge():
    method = make("blinking_bar", spacing_um=100, duty_cycle=0.20, offset_um=40)

    lit_rows = np.nonzero(method.spatial_mask()[:, 0])[0]
    assert lit_rows.min() == 40
    assert lit_rows.max() == 59


def test_blinking_bar_never_blinks_when_on_min_equals_period():
    # position 16: the limiting case, always on for the stimulation window
    method = make(
        "blinking_bar",
        spacing_um=100,
        duty_cycle=0.20,
        period_min=100,
        on_min=100,
        stop_min=22 * 60,
    )

    for minutes in (0, 20, 50, 99, 100, 700, 1319):
        assert lit_fraction(method, minutes) == pytest.approx(0.20, abs=0.01)
    assert lit_fraction(method, 22 * 60) == 0.0


def test_blinking_bar_survives_binning():
    method = make("blinking_bar", spacing_um=100, duty_cycle=0.20)
    method.update_binning(2)

    pattern = method.generate(_At(0))
    assert pattern.shape == (50, 5)
    assert pattern.mean() == pytest.approx(0.20, abs=0.02)


def test_global_pulse_twins_a_reference_bar():
    """
    The point of ``reference_bar_speed``: the train fires exactly when the
    reference bar reaches the middle of the field, minute for minute, with
    nothing measured off the rig.
    """
    bar = known_models["bar"](duty_cycle=0.2, bar_speed=1.0, period=100)
    pulse = known_models["global_pulse"](
        period_min=100, on_min=20, reference_bar_speed=1.0
    )
    # 250 um of field puts the centre at 125 um: a phase of 25 min, not 0
    camera = CameraProperties(ROI(0, 0, 4, 250), 1.0)
    for method in (bar, pulse):
        method.configure_system("exp.00", camera, make_experiment("exp.00"))

    assert pulse.effective_phase_min == pytest.approx(25.0)

    centre_row = int(pulse.center_um()[1])
    for minute in range(400):
        bar_at_centre = bar.generate(_At(minute))[centre_row, 0] > 0
        assert pulse.is_on(minute) == bar_at_centre, f"disagree at {minute} min"


def test_reference_bar_speed_follows_the_field_and_the_binning():
    def phase(height_px, pixel_um, binning=1):
        method = known_models["global_pulse"](
            period_min=100, on_min=20, reference_bar_speed=1.0
        )
        method.configure_system(
            "exp.00",
            CameraProperties(ROI(0, 0, 4, height_px), pixel_um),
            make_experiment("exp.00", stim_binning=binning),
        )
        return method.effective_phase_min

    # a bigger field means the centre is reached later
    assert phase(250, 1.0) == pytest.approx(25.0)
    assert phase(340, 1.0) == pytest.approx(70.0)
    # the same field described in binned pixels is the same field
    assert phase(500, 0.5) == pytest.approx(25.0)
    assert phase(250, 1.0, binning=2) == pytest.approx(25.0)


def test_global_pulse_rejects_a_stationary_reference():
    with pytest.raises(ValueError, match="reference_bar_speed must be non-zero"):
        make("global_pulse", reference_bar_speed=0)
