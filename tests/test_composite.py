"""The split pattern method: halves from the TOML, merged requirements, masks, the check."""

import shutil

import numpy as np
import pytest
from helpers import make_experiment
from test_check import RESOURCES, TOMLS
from test_dry_run import yml_experiment_dir

from pyclm import PatternMethod
from pyclm.check import check_directory
from pyclm.core.pattern_process import PatternProcess
from pyclm.core.patterns import ROI, CameraProperties, known_models
from pyclm.core.patterns.composite import SplitPattern
from pyclm.core.queues import AllQueues
from pyclm.schema import ExperimentConfig

SPLIT_TOML = """format_version = 1

[imaging]
exposure = 50

[channels]
presets = ["545"]

[stimulation]
exposure = 50

    [stimulation.config_groups]
    Channel = "DMD"

[segmentation]
method = "none"

[pattern]
method = "split"
boundary = 0.25

[pattern.left]
method = "move_in"
channel = "545"

[pattern.right]
method = "move_out"
channel = "545"
"""


class Const(PatternMethod):
    """A constant field, for telling the halves apart."""

    name = "const"

    def __init__(self, value=1.0, speed=0.0, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)
        self.speed = float(speed)

    def generate(self, context):
        return np.full(self.pattern_shape, self.value, np.float32)


REGISTRY = {**known_models, "const": Const}


def built(**kwargs) -> SplitPattern:
    method = SplitPattern(**kwargs)
    method.bind_registry(REGISTRY)
    method.configure_system(
        "exp.00", CameraProperties(ROI(0, 0, 40, 20), 1.0), make_experiment("exp.00")
    )
    return method


# ------------------------------------------------------------------ the TOML
def test_sub_tables_reach_the_method_as_kwargs():
    import tomllib

    cfg = ExperimentConfig.model_validate(tomllib.loads(SPLIT_TOML))
    assert cfg.pattern.method == "split"
    assert cfg.pattern.kwargs["left"] == {"method": "move_in", "channel": "545"}
    assert cfg.pattern.kwargs["boundary"] == 0.25
    exp = cfg.to_experiment("exp.00")
    assert exp.pattern.kwargs["right"]["method"] == "move_out"
    assert SplitPattern.nested_methods(cfg.pattern.kwargs) == [
        ("left", "move_in", {"channel": "545"}),
        ("right", "move_out", {"channel": "545"}),
    ]


def test_halves_must_be_a_matching_pair():
    with pytest.raises(ValueError, match="exactly"):
        SplitPattern(left={"method": "const"})
    with pytest.raises(ValueError, match="exactly"):
        SplitPattern(left={"method": "const"}, top={"method": "const"})
    with pytest.raises(ValueError, match="'method' key"):
        SplitPattern(left={"value": 1}, right={"method": "const"})
    with pytest.raises(ValueError, match="boundary"):
        SplitPattern(left={"method": "const"}, right={"method": "const"}, boundary=1.5)
    m = SplitPattern(left={"method": "const"}, right={"method": "nope"})
    with pytest.raises(ValueError, match="unknown pattern method 'nope'"):
        m.bind_registry(REGISTRY)


# ------------------------------------------------------------------ the pattern
def test_left_right_and_top_bottom_masks():
    m = built(
        left={"method": "const", "value": 1.0}, right={"method": "const", "value": 0.0}
    )
    pattern = m.generate(None)
    assert pattern.shape == (20, 40)
    assert pattern[:, :20].min() == 1.0
    assert pattern[:, 20:].max() == 0.0

    m = built(
        top={"method": "const", "value": 0.0},
        bottom={"method": "const", "value": 1.0},
        boundary=0.25,
    )
    pattern = m.generate(None)
    assert pattern[:5].max() == 0.0
    assert pattern[5:].min() == 1.0


def test_feather_blends_across_the_boundary():
    m = built(
        left={"method": "const", "value": 1.0},
        right={"method": "const", "value": 0.0},
        feather=10,
    )
    row = m.generate(None)[0]
    assert row[0] == 1.0
    assert row[-1] == 0.0
    ramp = row[15:25]
    assert np.all(np.diff(ramp) < 0)  # strictly decreasing across the feather
    assert abs(row[20] - 0.5) < 0.1


def test_requirements_are_the_union_of_the_halves():
    exp = make_experiment("exp.00")
    m = SplitPattern(
        left={"method": "move_in", "channel": "545"},
        right={"method": "move_out", "channel": "545"},
    )
    m.bind_registry(REGISTRY)
    reqs = m.initialize(exp)
    assert len(reqs) == 1
    [req] = reqs
    assert req.id == exp.channels["545"].channel_id
    assert req.needs_seg
    # the check reads the declared needs off the method
    assert [r[0] for r in m._requirements_list] == ["545", "545"]


def test_update_routes_to_the_halves():
    m = built(
        left={"method": "const", "speed": 1.0}, right={"method": "const", "speed": 2.0}
    )
    applied, refused = m.update(
        boundary=0.4, **{"left.speed": 5.0, "right.nope": 1, "feather": 3}
    )
    assert applied == {"boundary": 0.4, "feather": 3, "left.speed": 5.0}
    assert m.children["left"].speed == 5.0
    assert m.children["right"].speed == 2.0
    assert "right.nope" in refused


def test_pattern_process_builds_the_halves():
    pp = PatternProcess(AllQueues())
    pp.register_method(Const, "const")
    exp = make_experiment(
        "exp.00",
        pattern_method="split",
        pattern_kwargs={
            "left": {"method": "const", "value": 1.0},
            "right": {"method": "full_on"},
        },
    )
    pp.request_method(exp)
    pp.initialize(CameraProperties(ROI(0, 0, 40, 20), 1.0))
    pp.initialize_models()
    model = pp.models["exp.00"]
    assert isinstance(model, SplitPattern)
    assert sorted(model.children) == ["left", "right"]
    assert model.children["right"].pattern_shape == (20, 40)


# ------------------------------------------------------------------ the check
def test_check_validates_the_halves(tmp_path):
    for name in ("pyclm_config.toml", "PositionList.pos"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    (tmp_path / "bar10.toml").write_text(SPLIT_TOML)
    (tmp_path / "bar025.toml").write_text(SPLIT_TOML)
    report = check_directory(tmp_path)
    assert report.ok, report.text()

    (tmp_path / "bar10.toml").write_text(
        SPLIT_TOML.replace('method = "move_out"', 'method = "move_ot"').replace(
            'method = "move_in"\nchannel = "545"', 'method = "move_in"\nchanel = "545"'
        )
    )
    report = check_directory(tmp_path)
    text = report.text()
    assert "[pattern.right]: unknown pattern method 'move_ot'" in text
    assert "[pattern.left]: unknown argument 'chanel'" in text
    assert "did you mean 'channel'" in text


# ------------------------------------------------------------------ end to end
def test_dry_run_split_of_bar_and_full_on(yml_experiment_dir):
    """A run on the virtual microscope: left half a bar, right half fully lit."""
    import pyclm.io as pio
    from pyclm import run_pyclm

    config = yml_experiment_dir / "pyclm_config.toml"
    config.write_text(
        config.read_text().replace('format = "hdf5"', 'format = "ome-zarr"')
    )
    toml = yml_experiment_dir / "bar10.toml"
    text = toml.read_text()
    toml.write_text(
        text[: text.index("[pattern]")]
        + '[pattern]\nmethod = "split"\n\n[pattern.left]\nmethod = "bar"\n'
        'duty_cycle = 0.2\nbar_speed = 1.0\nperiod = 100\n\n[pattern.right]\nmethod = "full_on"\n'
    )
    run_pyclm(yml_experiment_dir, dry=True)

    with pio.open(yml_experiment_dir / "bar10.00.zarr") as exp:
        cam = exp.camera_pattern_at(exp.current_t)
    assert cam is not None
    _h, w = cam.shape
    assert cam[:, w // 2 :].min() == 255  # right half fully lit
    left = cam[:, : w // 2]
    assert 0 < (left > 0).mean() < 0.6  # the bar lights part of the left half
