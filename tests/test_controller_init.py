"""Controller.initialize guards: early output-file check, unused segmentation warning."""

import logging

import numpy as np
import pytest
from helpers import FakeImageSource, make_experiment, make_schedule

from pyclm import Controller

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def make_controller(**kwargs):
    return Controller(
        "unused.cfg",
        dry=True,
        dry_image_source=FakeImageSource((16, 16)),
        settle_time_s=0.0,
        **kwargs,
    )


def test_existing_output_is_rejected_before_models_load(tmp_path):
    controller = make_controller()
    schedule = make_schedule([make_experiment("exp.00")])
    (tmp_path / "exp.00.hdf5").write_bytes(b"")

    with pytest.raises(FileExistsError, match=r"exp\.00\.hdf5"):
        controller.initialize(schedule, (8, 8), IDENTITY, tmp_path)

    assert controller.pattern.models == {}
    assert not controller.outbox.writer.is_open


def test_unused_segmentation_method_warns(tmp_path, caplog):
    controller = make_controller()
    exp = make_experiment(
        "exp.00", segmentation_method="cellpose", pattern_method="full_on"
    )
    schedule = make_schedule([exp])

    try:
        with caplog.at_level(logging.WARNING):
            controller.initialize(schedule, (8, 8), IDENTITY, tmp_path)
    finally:
        controller.outbox.close_files()

    assert "does not request segmentation" in caplog.text
    assert controller.segmentation.models == {}
    assert "exp.00" in controller.pattern.models


def test_settle_time_reaches_microscope():
    controller = Controller(
        "unused.cfg", dry=True, dry_image_source=FakeImageSource(), settle_time_s=0.25
    )
    assert controller.microscope.settle_time_s == 0.25
