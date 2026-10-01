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


@pytest.mark.parametrize(
    ("storage_format", "suffix"), [("ome-zarr", ".zarr"), ("hdf5", ".hdf5")]
)
def test_existing_output_is_rejected_before_models_load(
    tmp_path, storage_format, suffix
):
    controller = make_controller(storage_format=storage_format)
    schedule = make_schedule([make_experiment("exp.00")])
    existing = tmp_path / f"exp.00{suffix}"
    if suffix == ".zarr":
        existing.mkdir()
    else:
        existing.write_bytes(b"")

    with pytest.raises(FileExistsError, match=rf"exp\.00\{suffix}"):
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


def test_shared_config_groups_are_applied_before_the_pixel_size_is_read(tmp_path):
    """The objective decides the pixel size; whatever the scope was left at, the
    run's presets are in force when the camera is asked."""
    from helpers import make_experiment, make_schedule

    from pyclm.core.experiments import ConfigGroup

    controller = make_controller()
    a = make_experiment("a.00")
    b = make_experiment("b.00")
    for exp in (a, b):
        for cfg in (*exp.channels.values(), exp.stimulation):
            cfg.update_config_groups([ConfigGroup("Objective", "20x")])
    a.channels["545"].update_config_groups([ConfigGroup("LightPath", "Fluor")])
    schedule = make_schedule([a, b])
    core = controller.core
    seen = {}
    real = core.getPixelSizeUm

    def spy():
        seen.update(core._config_groups)
        return real()

    core.getPixelSizeUm = spy
    controller.initialize(schedule, (12, 10), np.eye(2, 3, dtype=np.float32), tmp_path)
    assert seen.get("Objective") == "20x"  # shared by every channel: applied first
    assert "LightPath" not in seen  # only one channel sets it: not shared
