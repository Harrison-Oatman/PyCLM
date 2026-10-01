"""
The configuration schema (pyclm.schema): the same Experiment objects as the
hand parser produced, plus the mistakes it now catches with readable
messages.
"""

from pathlib import Path

import pytest

from pyclm.directories import experiment_from_toml, read_schedule
from pyclm.schema import (
    FORMAT_VERSION,
    ConfigError,
    ExperimentConfig,
    PyclmConfig,
    ScheduleConfig,
    field_table,
)

RESOURCES = Path(__file__).parent / "dry_run_resources"

FULL = """
format_version = 1
t_delay = 3
t_stop = 40

[config_groups]
Objective = "20x"

[device_properties]
"Cam-Gain" = 2

[imaging]
exposure = 50
every_t = 5
save = true
binning = 2

    [imaging.config_groups]
    LightPath = "Fluor"

[channels]
group = "Channel"
presets = ["545", "638"]

    [channels.638]
    exposure = 100
    every_t = 10

        [channels.638.config_groups]
        Laser = "High"

        [channels.638.device_properties]
        "Laser638-Power" = 0.5

[stimulation]
exposure = 200
every_t = 1
save = false

    [stimulation.config_groups]
    Channel = "DMD"

[segmentation]
method = "cellpose"
model = "cpsam"

[segmentation.nuclei]
method = "cellpose"
model = "nuclei"
save = false

[tracking]
method = "centroid"
segmentation = "nuclei"
max_distance_um = 12.5

[pattern]
method = "bar"
every_t = 2
duty_cycle = 0.3
"""


def write(tmp_path, text, name="exp.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def problems(model, tmp_path, text, name="exp.toml"):
    with pytest.raises(ConfigError) as info:
        model.from_file(write(tmp_path, text, name))
    return info.value.problems


def test_t_delay_after_a_table_header_is_explained(tmp_path):
    text = FULL.replace("t_delay = 3\nt_stop = 40\n", "") + "\nt_delay = 3\n"
    [problem] = problems(ExperimentConfig, tmp_path, text)
    assert problem.startswith(
        "[pattern]: t_delay must be written at the top of the file"
    )


# ---------------------------------------------------------------- builds
def test_full_experiment_builds_the_same_experiment_as_before(tmp_path):
    exp = experiment_from_toml(write(tmp_path, FULL), "exp.00")

    assert list(exp.channels) == ["545", "638"]
    c545, c638 = exp.channels["545"], exp.channels["638"]
    assert (c545.exposure, c545.every_t, c545.binning, c545.save) == (50, 5, 2, True)
    assert (c638.exposure, c638.every_t) == (100, 10)
    groups = {g.group: g.config for g in c638.get_config_groups()}
    assert groups == {
        "Objective": "20x",
        "LightPath": "Fluor",
        "Channel": "638",
        "Laser": "High",
    }
    props = {
        f"{d.device}-{d.property}": (d.value, d.type)
        for d in c638.get_device_properties()
    }
    assert props == {"Cam-Gain": (2, "int"), "Laser638-Power": (0.5, "float")}
    # the base tables reach the stimulation config too; the imaging ones do not
    stim_groups = {g.group: g.config for g in exp.stimulation.get_config_groups()}
    assert stim_groups == {"Objective": "20x", "Channel": "DMD"}
    assert (
        exp.stimulation.exposure,
        exp.stimulation.binning,
        exp.stimulation.save,
    ) == (200, 2, False)
    assert exp.segmentation.method_name == "cellpose"
    assert exp.segmentation.kwargs == {"model": "cpsam"}
    assert exp.segmentations["nuclei"].save is False
    assert exp.tracking.segmentation == "nuclei"
    assert exp.tracking.kwargs == {"max_distance_um": 12.5}
    assert exp.pattern.method_name == "bar"
    assert exp.pattern.every_t == 2
    assert exp.pattern.kwargs == {"duty_cycle": 0.3}
    assert (exp.t_delay, exp.t_stop) == (3, 40)


def test_lab_tomls_still_parse():
    for name in ("bar10", "bar025"):
        exp = experiment_from_toml(RESOURCES / "tomls" / f"{name}.toml", f"{name}.00")
        assert exp.channels["545"].every_t == 2
        assert exp.pattern.method_name == "bar"
    timing = read_schedule(RESOURCES / "test_schedule.toml")
    assert timing == {
        "t_count": 4,
        "t_interval": 1.0,
        "t_setup": 0.0,
        "t_between": 0.05,
    }


def test_minimal_experiment_uses_defaults(tmp_path):
    cfg = ExperimentConfig.from_file(
        write(
            tmp_path,
            '[channels]\ngroup = "Channel"\npresets = ["545"]\n\n[stimulation]\nexposure = 0\n\n'
            '[pattern]\nmethod = "full_on"\n',
        )
    )
    assert cfg.format_version == FORMAT_VERSION
    assert cfg.imaging.exposure == 10.0
    assert cfg.segmentation is None
    assert cfg.segmentation_names == []
    exp = cfg.to_experiment("x.00")
    assert exp.segmentation.method_name == "none"
    assert exp.tracking.method_name == "none"
    assert exp.stimulation.exposure == 0


# ------------------------------------------------------------- mistakes
def test_unknown_keys_are_errors_with_a_hint(tmp_path):
    text = FULL.replace("exposure = 50", "exposur = 50")
    found = problems(ExperimentConfig, tmp_path, text)
    assert len(found) == 1
    assert "unknown key 'exposur' in [imaging]" in found[0]
    assert "did you mean 'exposure'" in found[0]
    assert "allowed: binning, config_groups" in found[0]


def test_every_problem_in_a_file_is_reported_together(tmp_path):
    text = (
        FULL.replace("exposure = 50", "exposure = -5")
        .replace("every_t = 5", "every_t = 0")
        .replace('presets = ["545", "638"]', 'presets = ["545"]')
    )
    found = problems(ExperimentConfig, tmp_path, text)
    joined = "\n".join(found)
    assert "[imaging] exposure: must be greater than 0 (got -5)" in joined
    assert "[imaging] every_t: must be greater than or equal to 1 (got 0)" in joined
    assert "[channels.638] names a preset that is not in presets ['545']" in joined
    assert len(found) == 3


def test_missing_required_keys(tmp_path):
    found = problems(ExperimentConfig, tmp_path, '[channels]\ngroup = "Channel"\n')
    joined = "\n".join(found)
    assert "the file is missing the required key 'stimulation'" in joined
    assert "the file is missing the required key 'pattern'" in joined


def test_per_channel_binning_and_bad_property_keys_are_refused(tmp_path):
    text = FULL.replace("    exposure = 100\n", "    exposure = 100\n    binning = 1\n")
    assert any(
        "binning cannot differ between channels" in p
        for p in problems(ExperimentConfig, tmp_path, text)
    )
    text = FULL.replace('"Cam-Gain" = 2', '"CamGain" = 2')
    assert any(
        'must be written "Device-Property"' in p
        for p in problems(ExperimentConfig, tmp_path, text)
    )


def test_tracking_must_name_an_existing_segmentation(tmp_path):
    text = FULL.replace('segmentation = "nuclei"', 'segmentation = "cyto"')
    [problem] = problems(ExperimentConfig, tmp_path, text)
    assert "[tracking] segmentation = 'cyto'" in problem
    assert "['segmentation', 'nuclei']" in problem
    text = FULL.replace(
        '[segmentation]\nmethod = "cellpose"\nmodel = "cpsam"\n', ""
    ).replace(
        '[segmentation.nuclei]\nmethod = "cellpose"\nmodel = "nuclei"\nsave = false\n',
        "",
    )
    [problem] = problems(ExperimentConfig, tmp_path, text)
    assert "configures no segmentation" in problem


def test_save_output_alias_and_format_version(tmp_path):
    text = FULL.replace('model = "cpsam"', 'model = "cpsam"\nsave_output = false')
    cfg = ExperimentConfig.from_file(write(tmp_path, text))
    assert cfg.segmentation.save is False
    assert cfg.segmentation.kwargs == {"model": "cpsam"}
    text = FULL.replace("format_version = 1", "format_version = 99")
    [problem] = problems(ExperimentConfig, tmp_path, text)
    assert "newer than this PyCLM understands" in problem


def test_invalid_toml_and_missing_file(tmp_path):
    [problem] = problems(ExperimentConfig, tmp_path, "[imaging\nexposure = 1")
    assert "not valid TOML" in problem
    with pytest.raises(ConfigError, match="file not found"):
        ExperimentConfig.from_file(tmp_path / "nope.toml")


# ---------------------------------------------------- schedule and config
def test_schedule_requires_steps_and_interval(tmp_path):
    found = problems(
        ScheduleConfig, tmp_path, "[timing]\nsetup_time_seconds = 1\n", "schedule.toml"
    )
    joined = "\n".join(found)
    assert "missing the required key 'steps'" in joined
    assert "missing the required key 'interval_seconds'" in joined
    cfg = ScheduleConfig.from_file(
        write(tmp_path, "[timing]\nsteps = 3\ninterval_seconds = 30\n", "schedule.toml")
    )
    assert cfg.timing_kwargs() == {
        "t_count": 3,
        "t_interval": 30.0,
        "t_setup": 2.0,
        "t_between": 2.0,
    }


def test_pyclm_config_shape_and_output_choices(tmp_path):
    cfg = PyclmConfig.from_file(RESOURCES / "tomls" / "pyclm_config.toml")
    assert cfg.affine.shape == (2, 3)
    assert cfg.slm_shape == (1140, 912)
    assert cfg.output.format == "hdf5"
    assert cfg.focus_device == "ZDrive"

    text = 'config_path = "x.cfg"\naffine_transform = [[1, 0, 0]]\nslm_shape_h = 1\nslm_shape_w = 1\n'
    [problem] = problems(PyclmConfig, tmp_path, text, "pyclm_config.toml")
    assert "2 x 3 matrix" in problem
    text = (
        'config_path = "x.cfg"\naffine_transform = [[1, 0, 0], [0, 1, 0]]\nslm_shape_h = 1\n'
        'slm_shape_w = 1\n[output]\nformat = "tiff"\n'
    )
    [problem] = problems(PyclmConfig, tmp_path, text, "pyclm_config.toml")
    assert "[output] format" in problem
    assert "ome-zarr" in problem


def test_field_table_describes_every_key():
    rows = {r["key"]: r for r in field_table(ExperimentConfig)}
    assert rows["t_delay"]["default"] == "0"
    assert rows["stimulation"]["default"] == "required"
    assert rows["channels"]["default"] != "required"
    assert rows["t_stop"]["description"].startswith("stop after")
    timing = {
        r["key"]: r
        for r in field_table(ScheduleConfig.model_fields["timing"].annotation)
    }
    assert "> 0" in timing["interval_seconds"]["type"]


def test_stimulation_only_experiment():
    """No [channels] (or an empty presets list): the experiment only stimulates."""
    import tomllib

    text = (
        "format_version = 1"
        + "\n"
        + "[stimulation]"
        + "\n"
        + "exposure = 100"
        + "\n\n"
        + "[pattern]"
        + "\n"
        + 'method = "full_on"'
        + "\n"
    )
    cfg = ExperimentConfig.model_validate(tomllib.loads(text))
    assert cfg.channel_names == []
    exp = cfg.to_experiment("stim.00")
    assert exp.channels == {}
    with_empty = text.replace(
        "[stimulation]", "[channels]" + "\n" + "presets = []" + "\n\n" + "[stimulation]"
    )
    assert (
        ExperimentConfig.model_validate(tomllib.loads(with_empty)).channel_names == []
    )
