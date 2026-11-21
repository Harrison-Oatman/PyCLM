"""More than one SLM (DMD): ``[slm.<label>]`` in pyclm_config.toml, ``slm = "<label>"`` in ``[stimulation]``."""

from threading import Event
from uuid import uuid4

import numpy as np
import pytest
from helpers import FakeImageSource
from pydantic import ValidationError

from pyclm import cli
from pyclm.check import check_directory
from pyclm.core.datatypes import EventSLMPattern
from pyclm.core.events import UpdatePatternEvent
from pyclm.core.manager import SLMBuffer
from pyclm.core.microscope import MicroscopeProcess
from pyclm.core.queues import AllQueues
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore
from pyclm.schema import PyclmConfig, Stimulation

AFFINE = [[0.5, 0.0, 100.0], [0.0, 0.5, 200.0]]
SECOND_AFFINE = [[0.25, 0.0, 10.0], [0.0, 0.25, 20.0]]
SECOND = {"affine_transform": SECOND_AFFINE, "slm_shape_h": 500, "slm_shape_w": 400}
SECOND_TOML = (
    "\n[slm.second]\n"
    f"affine_transform = {SECOND_AFFINE}\n"
    "slm_shape_h = 500\nslm_shape_w = 400\n"
)


def config(**kw):
    base = {
        "config_path": "x.cfg",
        "affine_transform": AFFINE,
        "slm_shape_h": 1140,
        "slm_shape_w": 912,
    }
    return PyclmConfig.model_validate({**base, **kw})


def test_slm_tables_are_optional_and_validated():
    assert config().slm == {}
    assert config().slm_calibrations() == {}
    assert config().parallel_device_initialization is True
    cfg = config(slm={"second": SECOND}, parallel_device_initialization=False)
    shape, affine = cfg.slm_calibrations()["second"]
    assert shape == (500, 400)
    assert np.allclose(affine, SECOND_AFFINE)
    assert cfg.slm_shape == (1140, 912)  # the default SLM is untouched
    with pytest.raises(ValidationError):
        config(slm={"second": {**SECOND, "affine_transform": [[1, 0, 0]]}})
    with pytest.raises(ValidationError):
        config(slm={"second": {"affine_transform": SECOND_AFFINE}})


def test_stimulation_names_an_slm():
    assert Stimulation(exposure=10).slm is None
    assert Stimulation(exposure=10, slm="second").slm == "second"


def test_slm_buffer_uses_the_experiments_own_shape_and_affine():
    slm = SLMBuffer(AllQueues())
    slm.initialize(
        (1140, 912),
        np.array(AFFINE, np.float32),
        ["a", "b"],
        calibrations={"b": ((500, 400), np.array(SECOND_AFFINE, np.float32))},
    )
    pattern = np.ones((60, 80), np.float32)
    default = slm.pattern_to_slm(pattern, 1, "a")
    own = slm.pattern_to_slm(pattern, 1, "b")
    assert default.shape == (1140, 912)
    assert own.shape == (500, 400)
    assert np.array_equal(default, slm.pattern_to_slm(pattern, 1))  # no name: default
    ys, xs = np.nonzero(default)
    assert (xs.min(), ys.min()) == (100, 200)
    ys, xs = np.nonzero(own)
    assert (xs.min(), ys.min()) == (10, 20)
    assert xs.max() == pytest.approx(10 + 20, abs=1)
    # the blank patterns held before the first one arrives have each SLM's shape
    assert slm.slm_patterns["a"][1].shape == (1140, 912)
    assert slm.slm_patterns["b"][1].shape == (500, 400)


def make_microscope(slm_devices, shapes):
    aq = AllQueues()
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))
    core._slm_shapes.update(shapes)
    microscope = MicroscopeProcess(core, aq, stop_event=Event(), settle_time_s=0.0)
    microscope.slm_devices = slm_devices
    microscope.declare_slm()
    return aq, microscope, core


def test_microscope_sends_each_experiment_to_its_slm():
    aq, microscope, core = make_microscope({"b.00": "second"}, {"second": (500, 400)})
    # both SLMs are blanked when they are declared
    assert core._slm_images["SimulatedSLM"].shape == (1140, 900)
    assert core._slm_images["second"].shape == (500, 400)

    for name, shape, value in (("a.00", (1140, 900), 1), ("b.00", (500, 400), 2)):
        update = UpdatePatternEvent(name)
        aq.slm_to_microscope.put(
            EventSLMPattern(update.id, np.full(shape, value, np.uint8), str(uuid4()))
        )
        microscope.handle_update_pattern_event(update, slm_await_s=1.0)

    assert microscope._slm_for("a.00") == "SimulatedSLM"
    assert microscope._slm_for("b.00") == "second"
    assert core._slm_images["SimulatedSLM"].max() == 1
    assert core._slm_images["second"].max() == 2
    assert core._slm_images["second"].shape == (500, 400)

    microscope.blank_slm()
    assert core._slm_images["SimulatedSLM"].max() == 0
    assert core._slm_images["second"].max() == 0


def test_an_slm_table_naming_the_default_slm_is_refused():
    """MicroManager makes the SLM loaded last its default: an experiment without
    `slm` would then share the DMD of the one that names it."""
    aq = AllQueues()
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))
    microscope = MicroscopeProcess(core, aq, stop_event=Event(), settle_time_s=0.0)
    microscope.slm_devices = {"b.00": "SimulatedSLM"}
    microscope.experiment_names = ["a.00", "b.00"]
    with pytest.raises(RuntimeError, match="default SLM device"):
        microscope.declare_slm()
    # every experiment names it: nothing shares it by accident
    microscope.experiment_names = ["b.00"]
    microscope.declare_slm()


def test_an_unknown_slm_device_is_refused():
    class NoSuchDevice(SimulatedMicroscopeCore):
        def getSLMHeight(self, device):
            if device == "missing":
                raise RuntimeError('No device with label "missing"')
            return super().getSLMHeight(device)

    microscope = MicroscopeProcess(
        NoSuchDevice(FakeImageSource((16, 16))),
        AllQueues(),
        stop_event=Event(),
        settle_time_s=0.0,
    )
    microscope.slm_devices = {"b.00": "missing"}
    with pytest.raises(RuntimeError, match="is not an SLM device"):
        microscope.declare_slm()


def demo(tmp_path, slm_table=SECOND_TOML, label="second"):
    """The ``pyclm new --demo`` directory with its bar experiment on a second SLM."""
    target = tmp_path / "demo"
    assert cli.main(["new", str(target), "--demo"]) == 0
    lines = (target / "bar.toml").read_text().splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.strip().startswith("[stimulation]"))
    lines.insert(at + 1, f'slm = "{label}"')
    (target / "bar.toml").write_text("\n".join(lines) + "\n")
    with open(target / "pyclm_config.toml", "a") as f:
        f.write(slm_table)
    return target


def test_check_reports_slms_and_refuses_an_unknown_label(tmp_path):
    target = demo(tmp_path)
    report = check_directory(target)
    assert report.ok, report.text()
    text = report.text()
    assert "[slm.second]" in text
    assert "SLM 500 x 400" in text
    assert "used by bar" in text

    target = demo(tmp_path / "missing", slm_table="")
    report = check_directory(target)
    assert not report.ok
    assert any("has no [slm.second] table" in str(e) for e in report.errors)


def test_dry_run_records_each_experiment_with_its_own_slm(tmp_path):
    import pyclm.io as pio
    from pyclm import run_pyclm

    target = demo(tmp_path)
    (target / "schedule.toml").write_text(
        "[timing]\nsteps = 3\ninterval_seconds = 0.5\nsetup_time_seconds = 0.0\n"
        "time_between_positions = 0.05\n"
    )
    run_pyclm(target, dry=True)

    default_shape = PyclmConfig.from_file(target / "pyclm_config.toml").slm_shape
    with pio.open(target / "bar.00.zarr") as exp:
        assert exp.pattern_at(exp.current_t).shape == (500, 400)
        assert np.allclose(exp.affine_transform, SECOND_AFFINE)
        assert (exp.pattern_at(exp.current_t) > 0).any()  # the bar is on the DMD
    with pio.open(target / "cells.00.zarr") as exp:
        assert exp.pattern_at(exp.current_t).shape == tuple(default_shape)
        assert not np.allclose(exp.affine_transform, SECOND_AFFINE)
