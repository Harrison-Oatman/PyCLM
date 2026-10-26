"""Renamed pattern methods and keywords: old TOMLs keep working, check warns, --fix rewrites."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
from helpers import make_experiment

from pyclm.check import check_kwargs, migrate_pattern_kwargs
from pyclm.core.patterns import ROI, CameraProperties, known_models
from pyclm.core.patterns.zoo import ZooContext
from pyclm.fix import fix_text

RESOURCES = Path(__file__).parents[1] / "resources" / "tomls"


def make(name, shape=(60, 80), **kwargs):
    method = known_models[name](**kwargs)
    method.configure_system(
        "exp.00",
        CameraProperties(ROI(0, 0, shape[1], shape[0]), 1.0),
        make_experiment("exp.00"),
    )
    return method


def cells(shape=(60, 80)):
    seg = np.zeros(shape, np.int32)
    seg[5:20, 5:25] = 1
    seg[30:50, 40:70] = 2
    return seg


def frames(method, seg=None, times=(0, 600, 3600, 7300)):
    seg = cells(method.pattern_shape) if seg is None else seg
    return [
        np.asarray(method.render(ZooContext(seg.astype(float), seg, t)), float)
        for t in times
    ]


# old keywords -> new keywords, and the patterns must be identical
EQUIVALENT = [
    ("circle", {"rad": 20}, {"radius_um": 20}),
    (
        "bar",
        {"duty_cycle": 0.3, "bar_speed": 2.0, "period": 50},
        {"width_um": 15, "spacing_um": 50, "speed_um_per_min": 2.0},
    ),
    (
        "bar",
        {"duty_cycle": 0.25, "bar_speed": 0, "period": 40},
        {"width_um": 10, "spacing_um": 40, "speed_um_per_min": 0},
    ),
    (
        "bar_bounce",
        {"duty_cycle": 0.2, "bar_speed": 2.0, "period": 60, "t_loop": 30},
        {"width_um": 12, "spacing_um": 60, "speed_um_per_min": 2.0, "loop_min": 30},
    ),
    (
        "sawtooth",
        {"duty_cycle": 0.5, "bar_speed": 1.0, "period": 80, "inverse": True},
        {"width_um": 40, "spacing_um": 80, "speed_um_per_min": 1.0, "reverse": True},
    ),
    (
        "rings",
        {"duty_cycle": 0.2, "wave_speed": 5, "period": 60, "direction": -1},
        {"width_um": 12, "spacing_um": 60, "speed_um_per_min": -5},
    ),
    (
        "rotate_bar",
        {"num_bars": 4, "angular_velocity_rad": 0.05, "bar_width": 15},
        {"count": 4, "speed_deg_per_min": -np.degrees(0.05), "width_um": 15},
    ),
    (
        "blinking_bar",
        {"spacing_um": 80, "duty_cycle": 0.25, "period_min": 30, "on_min": 10},
        {"spacing_um": 80, "width_um": 20, "period_min": 30, "on_min": 10},
    ),
    (
        "global_pulse",
        {"period_min": 30, "on_min": 10, "reference_bar_speed": 2.0},
        {"period_min": 30, "on_min": 10, "reference_speed_um_per_min": 2.0},
    ),
    (
        "direction",
        {"sequence": "urdl", "period_h": 0.5},
        {"sequence": "urdl", "step_min": 30, "channel": "545"},
    ),
    (
        "binary_nucleus_clamp",
        {"clamp_target": 40},
        {"target": 40, "channel": "545"},
    ),
    ("embryo_inner", {"fraction_length": 0.2}, {"fraction": 0.2, "channel": "638"}),
]


@pytest.mark.parametrize(("name", "old", "new"), EQUIVALENT)
def test_old_keywords_give_the_same_pattern(name, old, new):
    current = getattr(known_models[name], "renamed_to", "") or name
    old_frames = frames(make(name, **old))
    new_frames = frames(make(current, **new))
    for a, b in zip(old_frames, new_frames, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize(("name", "old", "new"), EQUIVALENT)
def test_check_accepts_the_migrated_keywords(name, old, new):
    cls = known_models[name]
    migrated, notes, error = migrate_pattern_kwargs(cls, old)
    assert not error
    assert notes
    assert check_kwargs(cls, migrated) == []
    assert check_kwargs(known_models[cls.renamed_to or name], new) == []


def test_giving_the_old_and_the_new_name_is_an_error():
    with pytest.raises(ValueError, match="not both"):
        known_models["circle"](rad=5, radius_um=5)
    _new, _notes, error = migrate_pattern_kwargs(
        known_models["bar"], {"period": 5, "spacing_um": 5}
    )
    assert "not both" in error


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("rotate_ccw", "move_rotate"),
        ("direction", "move_sequence"),
        ("binary_nucleus_clamp", "nucleus_clamp"),
        ("centered_image", "target_image"),
        ("rotate_bar", "spokes"),
    ],
)
def test_renamed_methods_are_deprecated_aliases(old, new):
    alias = known_models[old]
    assert alias.renamed_to == new
    assert issubclass(alias, known_models[new])
    assert f'use method = "{new}"' in alias.deprecated
    assert not known_models[new].deprecated


def test_new_names_require_the_channel_the_old_ones_defaulted():
    with pytest.raises(AttributeError, match="channel"):
        known_models["nucleus_clamp"]()
    assert known_models["binary_nucleus_clamp"]().channel == "545"
    with pytest.raises(AttributeError, match="channel"):
        known_models["move_sequence"](sequence="u")
    assert known_models["direction"](sequence="u").channel == "545"


def test_move_rotate_reverse_lights_the_other_side():
    ccw = frames(make("move_rotate", channel="638"), times=(0,))[0]
    cw = frames(make("move_rotate", channel="638", reverse=True), times=(0,))[0]
    assert (ccw > 0).any()
    assert (cw > 0).any()
    assert not ((ccw > 0) & (cw > 0)).any()


def test_spokes_turn_counterclockwise_for_a_positive_speed():
    """After a quarter turn counterclockwise, a spoke pointing right points up."""
    method = make("spokes", shape=(101, 101), count=1, speed_deg_per_min=90, width_um=5)
    seg = np.zeros((101, 101), np.int32)
    right, up = frames(method, seg, times=(0, 60))
    assert right[50, 90]
    assert not right[10, 50]
    assert up[10, 50]
    assert not up[50, 90]


def test_every_method_is_dark_outside_start_and_stop():
    method = make("full_on", start_min=10, stop_min=20)
    before, during, after = frames(method, times=(9 * 60, 15 * 60, 20 * 60))
    assert before.max() == 0
    assert during.min() == 1
    assert after.max() == 0
    with pytest.raises(ValueError, match="must be after"):
        known_models["full_on"](start_min=10, stop_min=10)
    assert (
        check_kwargs(known_models["move_in"], {"channel": "638", "stop_min": 5}) == []
    )


def test_embryo_methods_work_on_a_non_square_field():
    seg = np.zeros((60, 100), np.int32)
    seg[20:40, 10:90] = 1  # long axis along the columns
    method = make("embryo_inner", shape=(60, 100), channel="638", fraction=0.2)
    [pattern] = frames(method, seg, times=(0,))
    lit_columns = np.flatnonzero(pattern.any(axis=0))
    assert pattern.shape == (60, 100)
    assert 40 <= lit_columns.min()
    assert lit_columns.max() <= 60
    outer = make("embryo_outer", shape=(60, 100), channel="638", fraction=0.2)
    [poles] = frames(outer, seg, times=(0,))
    assert poles[30, 12]
    assert poles[30, 87]
    assert not poles[30, 50]


# ------------------------------------------------------------------ --fix
OLD_TOML = """\
[pattern]
method = "direction"   # the per-cell sequence
sequence = "urdl"
period_h = 2           # hours per direction
voronoi = false
"""


def test_fix_renames_the_method_and_converts_the_keywords():
    text, changes, problems = fix_text(OLD_TOML, known_models)
    assert not problems
    assert 'method = "move_sequence"   # the per-cell sequence' in text
    assert "period_h" not in text
    table = tomllib.loads(text)["pattern"]
    assert table == {
        "method": "move_sequence",
        "sequence": "urdl",
        "step_min": 120,
        "voronoi": False,
        "channel": "545",
    }
    assert any("is now" in c for c in changes)
    # a second pass has nothing left to do
    again, changes, _problems = fix_text(text, known_models)
    assert again == text
    assert not changes


def test_fix_keeps_a_plain_rename_on_its_line_with_its_comment():
    text, _changes, _problems = fix_text(
        '[pattern]\nmethod = "circle"\nrad = 60   # um\n', known_models
    )
    assert text == '[pattern]\nmethod = "circle"\nradius_um = 60   # um\n'


def test_fix_updates_the_halves_of_a_split():
    old = (
        '[pattern]\nmethod = "split"\n\n[pattern.left]\nmethod = "bar"\n'
        "duty_cycle = 0.2\nbar_speed = 1.0\nperiod = 100\n\n"
        '[pattern.right]\nmethod = "full_on"\n'
    )
    text, _changes, problems = fix_text(old, known_models)
    assert not problems
    left = tomllib.loads(text)["pattern"]["left"]
    assert left == {
        "method": "bar",
        "speed_um_per_min": 1.0,
        "width_um": 20.0,
        "spacing_um": 100,
    }


def test_fix_leaves_a_value_spanning_lines_alone():
    old = '[pattern]\nmethod = "circle"\nrad = [\n  60\n]\n'
    text, changes, problems = fix_text(old, known_models)
    # a plain rename keeps the value as written, so it can still be done
    assert "radius_um = [" in text
    old = '[pattern]\nmethod = "bar"\nduty_cycle = 0.2\nperiod = [\n 1]\n'
    text, changes, problems = fix_text(old, known_models)
    assert text == old
    assert problems
    assert not changes


@pytest.mark.parametrize(
    "path", sorted(RESOURCES.rglob("*.toml")), ids=lambda p: p.name
)
def test_the_archived_example_tomls_still_build(path):
    """The resources/ TOMLs predate the renames; they must keep working."""
    data = tomllib.loads(path.read_text())
    if "pattern" not in data:
        pytest.skip("not an experiment TOML")
    table = dict(data["pattern"])
    cls = known_models.get(table.pop("method"))
    if cls is None:
        pytest.skip("a lab's own method")
    migrated, _notes, error = migrate_pattern_kwargs(cls, table)
    assert not error
    assert check_kwargs(cls, migrated) == []
