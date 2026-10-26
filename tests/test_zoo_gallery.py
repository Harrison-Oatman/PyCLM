"""The method zoo covers every built-in pattern method and builds."""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from helpers import make_experiment

from pyclm.core.patterns import ROI, CameraProperties, known_models
from pyclm.core.patterns.zoo import ZOO_GROUPS

DOCS = Path(__file__).parents[1] / "documentation"
sys.path.insert(0, str(DOCS / "ext"))

import zoo_gallery


@pytest.mark.parametrize("key", sorted(known_models))
def test_every_registered_method_is_in_the_zoo(key):
    meta = zoo_gallery.zoo_meta_for(key, known_models[key])
    assert meta is not None, f"{key!r} has no zoo_meta: give it a card or same_as"
    if meta.same_as:
        target = zoo_gallery.zoo_meta_for(meta.same_as, known_models[meta.same_as])
        assert target is not None
        assert not target.same_as
        assert meta.equivalent
    else:
        assert meta.group in {g[0] for g in ZOO_GROUPS}
        assert (DOCS / "zoo_sources" / f"{meta.source}.tif").is_file()


def test_the_gallery_builds_one_card_per_method(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("tifffile")

    out = tmp_path / "static" / "zoo"
    fragment = tmp_path / "_zoo_gallery.md"
    entries = zoo_gallery.build_gallery(DOCS / "zoo_sources", out, fragment, tmp_path)

    cards, aliases = zoo_gallery.collect(known_models)
    assert [e["key"] for e in entries] == [key for key, _ in cards]
    listed = {alias for pairs in aliases.values() for alias, _ in pairs}
    assert listed | {e["key"] for e in entries} == set(known_models)

    text = fragment.read_text()
    for e in entries:
        assert (out / e["thumb"]).is_file()
        assert (out / e["detail"]).is_file()
        assert f'href="#zoo-{e["key"].replace("_", "-")}"' in text
    for alias in listed:
        assert f'`"{alias}"`' in text


@pytest.mark.parametrize("key", sorted(known_models))
def test_the_toml_shown_is_valid_and_complete(key):
    meta = zoo_gallery.zoo_meta_for(key, known_models[key])
    if meta is None or meta.same_as:
        pytest.skip("no card")
    shown = {k: v for k, v in meta.kwargs.items() if not k.startswith("_")}
    parsed = tomllib.loads(zoo_gallery.toml_block(key, shown))["pattern"]
    assert parsed.pop("method") == key
    expected = {k: list(map(list, v)) if k == "pulses" else v for k, v in shown.items()}
    assert parsed == expected


def test_global_cycle_is_global_pulse_with_the_listed_arguments():
    def make(name, **kwargs):
        method = known_models[name](**kwargs)
        method.configure_system(
            "exp.00", CameraProperties(ROI(0, 0, 4, 4), 1.0), make_experiment("exp.00")
        )
        return method

    cycle = make("global_cycle", period_m=10)
    pulse = make("global_pulse", period_min=20, on_min=10)

    class At:
        def __init__(self, seconds):
            self.time = seconds

    for seconds in range(0, 3600 * 3, 37):
        assert np.array_equal(
            np.asarray(cycle.generate(At(seconds)), float),
            np.asarray(pulse.generate(At(seconds)), float),
        ), seconds


@pytest.mark.parametrize(
    ("name", "kwargs", "direction_kwargs"),
    [
        ("move_down", {}, {"sequence": "d"}),
        # fb_bounce reverses every half-loop: down, then up, then down again
        ("fb_bounce", {"t_loop": 60}, {"sequence": "du", "step_min": 30}),
    ],
)
def test_per_cell_aliases_match_move_sequence(name, kwargs, direction_kwargs):
    from pyclm.core.patterns.zoo import ZooContext

    seg = np.zeros((40, 40), np.int32)
    seg[5:15, 5:15] = 1
    seg[22:36, 20:30] = 2

    def make(key, **kw):
        method = known_models[key](channel="638", **kw)
        method.pattern_shape = seg.shape
        return method

    alias = make(name, **kwargs)
    general = make("move_sequence", voronoi=False, gradient=False, **direction_kwargs)
    for minutes in (1, 29, 31, 59, 61, 89, 91, 121):
        context = ZooContext(seg.astype(float), seg, time_seconds=minutes * 60)
        assert np.array_equal(
            np.asarray(alias.generate(context), float),
            np.asarray(general.generate(context), float),
        ), minutes


@pytest.mark.parametrize(
    ("letter", "lit_rows"), [("u", slice(0, 5)), ("d", slice(5, 10))]
)
def test_move_sequence_lights_the_side_cells_should_move_to(letter, lit_rows):
    """Cells move towards the light; rows grow down the image."""
    from pyclm.core.patterns.zoo import ZooContext

    seg = np.zeros((10, 10), np.int32)
    seg[:, 2:8] = 1
    method = known_models["move_sequence"](
        sequence=letter, channel="638", voronoi=False, gradient=False
    )
    method.pattern_shape = seg.shape
    pattern = np.asarray(method.generate(ZooContext(seg.astype(float), seg)), float)
    lit = np.zeros_like(pattern, bool)
    lit[lit_rows, 2:8] = True  # the centroid is at row 4.5
    assert np.array_equal(pattern > 0, lit)


def test_gradient_mode_keeps_a_one_pixel_cell_dark():
    from pyclm.core.patterns.zoo import ZooContext

    seg = np.zeros((20, 20), np.int32)
    seg[3, 3] = 1  # no pixel ahead of its centroid
    seg[10:16, 10:16] = 2
    method = known_models["move_out"](channel="638", gradient=True)
    method.pattern_shape = seg.shape
    pattern = np.asarray(method.generate(ZooContext(seg.astype(float), seg)), float)
    assert not np.isnan(pattern).any()
    assert pattern[3, 3] == 0
    assert pattern.max() == 1
