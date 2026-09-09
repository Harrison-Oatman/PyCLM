"""
pyclm check: findings for a directory that is fine, for the mistakes a new
user makes, against a MicroManager .cfg, and the run refusing to start on
errors.
"""

import shutil
from pathlib import Path

import pytest

from pyclm.check import CheckFailed, check_directory, check_kwargs
from pyclm.core.patterns import PatternMethod
from pyclm.core.patterns.bar_patterns import BarPattern
from pyclm.core.segmentation.cellpose_segmentation import CellposeSegmentationMethod
from pyclm.core.tracking import CentroidTracker
from pyclm.mmconfig import MMConfig

RESOURCES = Path(__file__).parent / "dry_run_resources"
TOMLS = RESOURCES / "tomls"

MM_CFG = """# a MicroManager configuration, the parts pyclm reads
Device,Core,Core,Core
Device,Camera,DemoCamera,DCam
Device,Sola,DemoCamera,DLightPath
Property,Camera,Binning,1
ConfigGroup,System,Startup,Camera,Binning,1
ConfigGroup,Objective,1-Plan Apo LmbdD0.80 20x,Core,Focus,ZDrive
ConfigGroup,Laser-Intensity,50%,Sola,Power,50
ConfigGroup,Laser-Intensity,AllOff,Sola,Power,0
ConfigGroup,LightPath,Fluor,Core,Shutter,Shutter
ConfigGroup,LightPath,DMD,Core,Shutter,Shutter
ConfigGroup,Channel,545,Sola,Color,Green
ConfigGroup,Channel,DMD,Sola,Color,Red
ConfigGroup,Sola,30,Sola,Power,30
"""


@pytest.fixture
def good_dir(tmp_path):
    for name in ("bar10.toml", "bar025.toml", "pyclm_config.toml", "PositionList.pos"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    return tmp_path


def messages(report, level=None):
    return [str(f) for f in report.findings if level is None or f.level == level]


# ------------------------------------------------------------------ happy
def test_a_correct_directory_has_no_errors(good_dir):
    report = check_directory(good_dir)
    assert report.ok, report.text()
    text = report.text()
    assert "4 timepoints every 1 s" in text
    assert "PositionList.pos: 2 position(s): bar10.pos1, bar025.pos1" in text
    assert "frames over 4 timepoints" in text
    assert "MicroManager configuration not found" in text
    # the test schedule is deliberately faster than the simulated settle time
    [warning] = report.warnings
    assert warning.where == "[timing]"


# --------------------------------------------------------------- mistakes
def test_misspelled_method_argument_is_an_error_with_a_hint(good_dir):
    toml = good_dir / "bar10.toml"
    toml.write_text(toml.read_text().replace("bar_speed = 1.0", "bar_sped = 1.0"))
    report = check_directory(good_dir)
    [error] = messages(report, "error")
    assert "bar10.toml [pattern]: unknown argument 'bar_sped' for method 'bar'" in error
    assert "did you mean 'bar_speed'" in error
    assert "accepted: bar_speed, duty_cycle, period" in error


def test_unknown_key_in_a_table_is_an_error(good_dir):
    toml = good_dir / "bar10.toml"
    toml.write_text(toml.read_text().replace("every_t = 2", "evry_t = 2", 1))
    report = check_directory(good_dir)
    [error] = messages(report, "error")
    assert "unknown key 'evry_t' in [imaging]" in error
    assert "did you mean 'every_t'" in error


def test_position_without_a_toml_and_toml_without_a_position(good_dir):
    (good_dir / "bar025.toml").rename(good_dir / "other.toml")
    report = check_directory(good_dir)
    errors = messages(report, "error")
    assert any(
        "PositionList.pos bar025.pos1: no experiment file 'bar025.toml'" in e
        for e in errors
    )
    assert any(
        "other.toml: not used by any position" in w for w in messages(report, "warning")
    )


def test_requirements_are_checked_against_the_tables(good_dir):
    toml = good_dir / "bar10.toml"
    text = toml.read_text()
    text = (
        text[: text.index("[pattern]")]
        + '[pattern]\nmethod = "move_out"\nchannel = "638"\n'
    )
    toml.write_text(text)
    report = check_directory(good_dir)
    errors = "\n".join(messages(report, "error"))
    assert "needs channel '638', which is not in [channels] presets ['545']" in errors
    assert "needs a segmentation but the file has no [segmentation] table" in errors

    text = text.replace('channel = "638"', 'channel = "545"\ntracks = true')
    text = text.replace("[pattern]", '[segmentation]\nmethod = "cellpose"\n\n[pattern]')
    toml.write_text(text)
    report = check_directory(good_dir)
    errors = "\n".join(messages(report, "error"))
    assert "needs tracks but the file has no [tracking] table" in errors


def test_unused_segmentation_and_tracking_are_warnings(good_dir):
    toml = good_dir / "bar10.toml"
    toml.write_text(
        toml.read_text().replace(
            "[pattern]",
            '[segmentation]\nmethod = "cellpose"\n\n[tracking]\nmethod = "centroid"\n\n[pattern]',
        )
    )
    report = check_directory(good_dir)
    assert report.ok
    warnings = "\n".join(messages(report, "warning"))
    assert (
        "bar10.toml [segmentation]: configured but the pattern method 'bar' does not ask for it"
        in warnings
    )
    assert (
        "bar10.toml [tracking]: configured but the pattern method 'bar' does not ask for tracks"
        in warnings
    )


def test_unknown_methods_and_existing_outputs(good_dir):
    toml = good_dir / "bar10.toml"
    toml.write_text(toml.read_text().replace('method = "bar"', 'method = "barr"'))
    (good_dir / "bar025.pos1.zarr").mkdir()
    report = check_directory(good_dir)
    errors = "\n".join(messages(report, "error"))
    assert "unknown pattern method 'barr'" in errors
    assert "bar025.pos1.zarr: output already exists" in errors


def test_missing_schedule_and_config(tmp_path):
    shutil.copy(TOMLS / "bar10.toml", tmp_path / "bar10.toml")
    report = check_directory(tmp_path, config_path=tmp_path / "nope.toml")
    errors = "\n".join(messages(report, "error"))
    assert "nope.toml: file not found" in errors
    assert "schedule.toml: file not found" in errors
    assert "positions: no PositionList.pos or multipoints.xml" in errors


def test_timing_budget_warning(good_dir):
    (good_dir / "schedule.toml").write_text(
        "[timing]\nsteps = 3\ninterval_seconds = 0.01\nsetup_time_seconds = 0\n"
    )
    report = check_directory(good_dir)
    assert report.ok
    [warning] = [w for w in messages(report, "warning") if "[timing]" in w]
    assert (
        "3 of 3 timepoints are estimated to take longer than the 0.01 s interval"
        in warning
    )


# ------------------------------------------------------------- MM config
def test_mm_config_is_read_from_text(tmp_path):
    cfg_path = tmp_path / "mm.cfg"
    cfg_path.write_text(MM_CFG)
    mm = MMConfig.from_file(cfg_path)
    assert mm.has_group("Channel")
    assert mm.presets("Channel") == ["545", "DMD"]
    assert mm.has_preset("LightPath", "DMD")
    assert not mm.has_preset("LightPath", "Off")
    assert mm.has_device("Camera")
    assert mm.has_property("Camera", "Binning") is True
    assert mm.has_property("Camera", "Gain") is None
    assert mm.has_property("Nope", "Gain") is False
    assert "6 config groups" in mm.summary()


def test_presets_and_devices_are_checked_against_the_cfg(good_dir):
    cfg_path = good_dir / "mm.cfg"
    cfg_path.write_text(MM_CFG)
    report = check_directory(good_dir, mm_config=cfg_path)
    assert report.ok, report.text()

    toml = good_dir / "bar10.toml"
    text = toml.read_text().replace('LightPath = "Fluor"', 'LightPath = "Fluoro"')
    text = text.replace(
        "[device_properties]\n",
        '[device_properties]\n"Camera-Gain" = 2\n"Laser-Power" = 1\n',
    )
    toml.write_text(text)
    report = check_directory(good_dir, mm_config=cfg_path)
    errors = "\n".join(messages(report, "error"))
    warnings = "\n".join(messages(report, "warning"))
    assert (
        "preset 'Fluoro' is not in config group 'LightPath' (presets: DMD, Fluor)"
        in errors
    )
    assert (
        "device 'Laser' ('Laser-Power') is not in the MicroManager configuration"
        in errors
    )
    assert "property 'Gain' of device 'Camera' is not listed" in warnings


# -------------------------------------------------------------- signatures
def test_check_kwargs_against_signatures():
    assert check_kwargs(BarPattern, {"duty_cycle": 0.1, "period": 5}) == []
    [problem] = check_kwargs(BarPattern, {"perio": 5})
    assert (
        "unknown argument 'perio' for method 'bar' (did you mean 'period'?)" in problem
    )
    assert "extra keys are silently ignored" in problem
    assert (
        check_kwargs(CellposeSegmentationMethod, {"model": "cpsam"}, "segmentation")
        == []
    )
    assert check_kwargs(CentroidTracker, {"max_distance_um": 5}, "tracking") == []
    # a pattern method's `channel` is the user's to give; a tracker's is supplied
    [problem] = check_kwargs(CentroidTracker, {"channel": "545"}, "tracking")
    assert problem.startswith("unknown argument 'channel'")

    class Needs(PatternMethod):
        name = "needs"

        def __init__(self, channel, gain=1.0, **kwargs):
            super().__init__(**kwargs)

    [problem] = check_kwargs(Needs, {"gain": 2})
    assert "requires the argument 'channel'" in problem


# ------------------------------------------------------------------- run
def test_run_refuses_to_start_on_errors_unless_forced(good_dir):
    from pyclm.run_pyclm import run_pyclm

    toml = good_dir / "bar10.toml"
    toml.write_text(toml.read_text().replace("bar_speed = 1.0", "bar_sped = 1.0"))
    with pytest.raises(CheckFailed, match="1 error"):
        run_pyclm(good_dir, dry=True)
    assert not list(good_dir.glob("*.hdf5"))
