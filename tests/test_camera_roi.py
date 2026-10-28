"""The camera ROI: config validation, affine composition, the virtual camera, the check."""

import shutil

import cv2
import numpy as np
import pytest
from helpers import FakeImageSource
from pydantic import ValidationError
from test_check import RESOURCES, TOMLS

from pyclm.check import check_directory
from pyclm.core.manager import SLMBuffer, compose_affine
from pyclm.core.queues import AllQueues
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore
from pyclm.schema import PyclmConfig

AFFINE = [[0.5, 0.0, 100.0], [0.0, 0.5, 200.0]]


def config(**kw):
    base = {
        "config_path": "x.cfg",
        "affine_transform": AFFINE,
        "slm_shape_h": 1140,
        "slm_shape_w": 912,
    }
    return PyclmConfig.model_validate({**base, **kw})


def test_camera_roi_is_validated():
    assert config().camera_roi is None
    assert config(camera_roi=[10, 20, 300, 400]).camera_roi == [10, 20, 300, 400]
    for bad in ([0, 0, 0, 10], [0, 0, 10], [-1, 0, 10, 10]):
        with pytest.raises(ValidationError):
            config(camera_roi=bad)


def test_dmd_footprint_is_the_inverse_affine_of_the_dmd_rectangle():
    # x: (0 - 100) / 0.5 .. (912 - 100) / 0.5, y: (0 - 200) / 0.5 .. (1140 - 200) / 0.5
    assert config().dmd_footprint() == (-200, -400, 1824, 2280)


def test_compose_affine_equals_warping_the_pattern_embedded_in_the_full_frame():
    rng = np.random.default_rng(0)
    pattern = (rng.random((60, 80)) > 0.7).astype(np.uint8) * 255
    ox, oy = 300, 120
    full = np.zeros((600, 800), np.uint8)
    full[oy : oy + 60, ox : ox + 80] = pattern
    at = np.array(AFFINE, np.float32)

    expected = cv2.warpAffine(full, at, (912, 1140))
    got = cv2.warpAffine(pattern, compose_affine(at, (ox, oy), 1), (912, 1140))

    assert np.array_equal(got > 127, expected > 127)
    # binning scales the linear part after the offset is composed; binned
    # pixel p is centered on unbinned 2p + 0.5
    at2 = compose_affine(at, (ox, oy), 2)
    assert np.allclose(at2[:, :2], at[:, :2] * 2)
    assert np.allclose(at2[:, 2], at[:, 2] + at[:, :2] @ [ox + 0.5, oy + 0.5])


@pytest.mark.parametrize("binning", [2, 3, 4])
def test_a_binned_pattern_lands_where_its_unbinned_pixels_do(binning):
    """A binned pixel lights the DMD where the block of unbinned pixels it
    covers would: centered, not shifted by (binning - 1) / 2 unbinned pixels.
    (Session 3 measured the old shift: 0.25 binned px at binning 2.)"""
    at = np.array([[0.31, 0.02, 100.0], [-0.01, 0.58, 200.0]], np.float32)
    ox, oy = 300, 120
    for p in ([0, 0], [7, 3], [50, 41]):
        block = [binning * p[0] + ox, binning * p[1] + oy]
        center = np.mean(
            [
                [block[0] + i, block[1] + j]
                for i in range(binning)
                for j in range(binning)
            ],
            axis=0,
        )
        want = at[:, :2] @ center + at[:, 2]
        composed = compose_affine(at, (ox, oy), binning)
        got = composed[:, :2] @ np.array(p, np.float32) + composed[:, 2]
        assert np.allclose(got, want, atol=1e-3)


def test_slm_buffer_applies_the_roi_offset():
    slm = SLMBuffer(AllQueues())
    slm.initialize(
        (1140, 912), np.array(AFFINE, np.float32), ["a"], roi=(300, 120, 80, 60)
    )
    pattern = np.zeros((60, 80), np.float32)
    pattern[:, :] = 1.0
    dmd = slm.pattern_to_slm(pattern, 1)
    ys, xs = np.nonzero(dmd)
    # the ROI's top-left (300, 120) lands at (0.5 * 300 + 100, 0.5 * 120 + 200) = (250, 260)
    assert xs.min() == 250
    assert ys.min() == 260
    assert xs.max() == pytest.approx(250 + 40, abs=1)
    assert ys.max() == pytest.approx(260 + 30, abs=1)


def test_simulated_core_set_roi_crops_the_frame_in_unbinned_pixels():
    core = SimulatedMicroscopeCore(FakeImageSource((100, 120)))
    assert core.getROI() == (0, 0, 480, 400)
    core.setROI(40, 80, 200, 120)
    assert core.getROI() == (40, 80, 200, 120)
    core.snapImage()
    assert core.getImage().shape == (30, 50)
    with pytest.raises(ValueError, match="exceeds the virtual camera"):
        core.setROI(0, 0, 4000, 4000)


def test_check_reports_the_dmd_footprint_and_an_roi_beyond_it(tmp_path):
    for name in ("bar10.toml", "bar025.toml", "PositionList.pos"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    cfg = (TOMLS / "pyclm_config.toml").read_text()
    (tmp_path / "pyclm_config.toml").write_text(cfg)
    text = check_directory(tmp_path).text()
    assert "camera_roi: not set" in text
    assert "the DMD covers x" in text

    # the key must sit above the [output] table
    (tmp_path / "pyclm_config.toml").write_text(
        cfg.replace(
            "config_path", "camera_roi = [0, 0, 100000, 100000]\nconfig_path", 1
        )
    )
    report = check_directory(tmp_path)
    assert any(
        "extends beyond the region the DMD can light" in str(w) for w in report.warnings
    )


def test_suggested_camera_roi_is_clipped_to_the_sensor():
    """A DMD footprint larger than the sensor (session 3: x -64..1907, y -478..2674
    on a 2048 x 2048 camera) gives an ROI inside the sensor."""
    from pyclm.schema import suggest_camera_roi

    roi, unreached = suggest_camera_roi((-64, -478, 1971, 3152), (2048, 2048))
    assert roi == [0, 0, 1907, 2048]
    assert unreached == ["right"]
    # without the sensor only the origin is clipped, and the size shrinks with it
    roi, _ = suggest_camera_roi((-64, -478, 1971, 3152))
    assert roi == [0, 0, 1907, 2674]


def test_slm_buffer_holds_handshakes_until_the_first_pattern():
    """Before an experiment's first pattern exists the buffer does not answer
    with the blank placeholder (the microscope would skip the stimulation)."""
    from pyclm.core.datatypes import CameraPattern
    from pyclm.core.events import UpdatePatternEvent
    from pyclm.core.messages import UpdatePatternEventMessage

    aq = AllQueues()
    slm = SLMBuffer(aq)
    slm.initialize((1140, 912), np.array(AFFINE, np.float32), ["a"])
    event = UpdatePatternEvent("a")
    slm.handle_message(UpdatePatternEventMessage(event))
    assert aq.slm_to_microscope.empty()

    slm.handle_data(CameraPattern("a", np.ones((60, 80), np.float32)))
    reply = aq.slm_to_microscope.get_nowait()
    assert reply.event_id == event.id
    assert reply.pattern.any()

    # once a pattern exists, handshakes are answered at once
    later = UpdatePatternEvent("a")
    slm.handle_message(UpdatePatternEventMessage(later))
    assert aq.slm_to_microscope.get_nowait().event_id == later.id


def test_slm_buffer_stops_holding_after_the_wait():
    from pyclm.core.events import UpdatePatternEvent
    from pyclm.core.messages import UpdatePatternEventMessage

    aq = AllQueues()
    slm = SLMBuffer(aq)
    slm.first_pattern_wait_s = -1.0
    slm.initialize((1140, 912), np.array(AFFINE, np.float32), ["a"])
    first, second = UpdatePatternEvent("a"), UpdatePatternEvent("a")
    slm.handle_message(UpdatePatternEventMessage(first))
    slm.handle_message(UpdatePatternEventMessage(second))
    ids = [aq.slm_to_microscope.get_nowait().event_id for _ in range(2)]
    assert ids == [first.id, second.id]
