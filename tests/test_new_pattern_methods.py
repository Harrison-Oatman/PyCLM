"""The pattern methods added in 2.0: static, timed, per-cell, per-nucleus, adaptive, diagnostic."""

import numpy as np
import pytest
from helpers import make_experiment

from pyclm.check import check_kwargs
from pyclm.core.measure import Regions
from pyclm.core.patterns import ROI, CameraProperties, known_models
from pyclm.core.patterns.dose import BAYER, deliver
from pyclm.core.tracking import TrackRow, Tracks


def make(name, shape=(60, 80), pixel_um=1.0, camera_to_stage=None, **kwargs):
    method = known_models[name](**kwargs)
    method.bind_registry(known_models)
    method.configure_system(
        "exp.00",
        CameraProperties(ROI(0, 0, shape[1], shape[0]), pixel_um, camera_to_stage),
        make_experiment("exp.00"),
    )
    return method


class Context:
    """A PatternContext stand-in: fixed frames, a history, a position, recorded requests."""

    def __init__(self, raw=None, seg=None, time=0.0, history=None, position=None):
        self._raw = raw
        self._seg = seg
        self.time = float(time)
        self._history = history
        self._position = position
        self.requests = []
        self.exposure_ms = 10.0

    def raw(self, channel):
        return self._raw

    def segmentation(self, channel, name="segmentation"):
        return self._seg

    def regions(self, channel, name="segmentation"):
        return Regions(self._seg)

    def tracks(self, channel):
        if self._seg is None:
            return None
        regions = Regions(self._seg)
        rows = [
            TrackRow(int(i), int(i), float(c[0]), float(c[1]), int(a), 0)
            for i, c, a in zip(
                regions.ids, regions.centroids(), regions.areas(), strict=True
            )
        ]
        return Tracks(self._seg, rows)

    def history(self, channel, kind="seg", n=None):
        frames = self._history if self._history is not None else [self._raw]
        return frames if n is None else frames[-n:]

    def stim_raw(self):
        return self._raw

    def position(self):
        return self._position

    def settings(self, channel):
        return {"exposure_ms": self.exposure_ms}

    def set_exposure(self, channel, ms):
        self.requests.append(("exposure", channel, ms))

    def set_config(self, channel, group, preset):
        self.requests.append(("config", channel, group, preset))

    def set_property(self, channel, device, prop, value):
        self.requests.append(("property", channel, device, prop, value))

    def set_position(self, x=None, y=None, z=None, pfs_offset=None):
        self.requests.append(("position", x, y))


def two_cells(shape=(60, 80)):
    seg = np.zeros(shape, np.int32)
    seg[5:15, 5:25] = 1  # small, top left
    seg[30:50, 40:70] = 2  # large, lower right
    return seg


def lit(method, context):
    return np.asarray(method.render(context), float)


# ------------------------------------------------------------------ dose
def test_dose_modes():
    dose = np.full((16, 16), 0.25, np.float32)
    assert np.array_equal(deliver(dose, "gray"), dose)
    dithered = deliver(dose, "dither")
    assert set(np.unique(dithered)) == {0.0, 1.0}
    assert dithered.mean() == pytest.approx(0.25)
    assert deliver(np.ones((4, 4)), "dither").min() == 1
    assert deliver(np.zeros((4, 4)), "dither").max() == 0
    # duty: on for the first quarter of each 10 min period
    on = [deliver(dose, "duty", t, 10.0).max() for t in np.arange(0, 20, 0.5)]
    assert np.mean(on) == pytest.approx(0.25)
    assert on[0] == 1
    assert on[5] == 0
    assert BAYER.shape == (8, 8)


# ------------------------------------------------------------------ static
def test_rectangle_offset_and_angle():
    m = make("rectangle", shape=(101, 101), width_um=40, height_um=10, y_um=-20)
    p = lit(m, Context())
    assert p[30, 50]
    assert p[30, 31]
    assert not p[30, 25]
    assert not p[50, 50]  # moved up by 20 um
    turned = make(
        "rectangle", shape=(101, 101), width_um=40, height_um=10, angle_deg=90
    )
    q = lit(turned, Context())
    assert q[35, 50]
    assert q[65, 50]
    assert not q[50, 35]
    with pytest.raises(ValueError, match="width_um"):
        known_models["rectangle"](height_um=5)


def test_dose_grid_blocks_doses_and_spots():
    m = make("dose_grid", shape=(100, 100), doses=[0.0, 0.5, 1.0], gap_um=10)
    p = lit(m, Context())
    assert p[25, 25] == 0
    assert p[25, 75] == 0.5
    assert p[75, 25] == 1.0
    assert p[75, 75] == 0  # the list is cycled
    assert p[50, 25] == 0
    assert p[2, 25] == 0
    spots = make("dose_grid", shape=(100, 100), spot_um=20, gap_um=0)
    s = lit(spots, Context())
    assert s[25, 25] == 1
    assert s[25, 40] == 0
    assert s.sum() == pytest.approx(4 * np.pi * 10**2, rel=0.1)
    dithered = make(
        "dose_grid", shape=(100, 100), doses=[0.5], gap_um=10, dose_mode="dither"
    )
    assert set(np.unique(lit(dithered, Context()))) == {0.0, 1.0}
    with pytest.raises(ValueError, match="doses"):
        known_models["dose_grid"](doses=[2])
    with pytest.raises(ValueError, match="dose_mode"):
        known_models["dose_grid"](dose_mode="pwm")


def test_image_mask_center_stretch_and_threshold(tmp_path):
    import tifffile

    img = np.zeros((10, 20), np.uint16)
    img[:, 10:] = 1000
    img[:, 5:10] = 500
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, img)
    centered = lit(make("image_mask", shape=(30, 40), path=str(path)), Context())
    # the 10 x 20 image sits in rows 10-19, columns 10-29
    assert centered[15, 25] == 1.0
    assert centered[15, 17] == 0.5
    assert centered[15, 35] == 0
    assert centered[5, 25] == 0
    stretched = lit(
        make("image_mask", shape=(30, 40), path=str(path), fit="stretch"), Context()
    )
    assert stretched[1, 39] == pytest.approx(1.0)
    binary = lit(
        make("image_mask", shape=(30, 40), path=str(path), threshold=0.6), Context()
    )
    assert binary[15, 25] == 1
    assert binary[15, 17] == 0
    with pytest.raises(ValueError, match="fit"):
        known_models["image_mask"](path=str(path), fit="tile")


# ------------------------------------------------------------------ timed
def test_pulsed_gates_the_inner_method_and_keeps_it_running():
    m = make(
        "pulsed",
        period_min=30,
        on_min=10,
        phase_min=5,
        inner={"method": "circle", "radius_um": 10},
    )
    circle = lit(make("circle", radius_um=10), Context())
    assert lit(m, Context(time=0)).max() == 0
    assert np.array_equal(lit(m, Context(time=6 * 60)), circle)
    assert lit(m, Context(time=16 * 60)).max() == 0
    assert np.array_equal(lit(m, Context(time=36 * 60)), circle)
    with pytest.raises(ValueError, match=r"\[pattern.inner\]"):
        known_models["pulsed"](period_min=10, on_min=5)
    # pulsed around full_on is global_pulse
    pulse = make("global_pulse", period_min=30, on_min=10)
    wrapped = make("pulsed", period_min=30, on_min=10, inner={"method": "full_on"})
    for t in range(0, 7200, 90):
        assert np.array_equal(
            lit(pulse, Context(time=t)), lit(wrapped, Context(time=t))
        )


def test_chirp_sweeps_the_period():
    for sweep in ("log", "linear"):
        m = make(
            "chirp", period_start_min=40, period_end_min=10, sweep_min=200, sweep=sweep
        )
        assert m.period_at(0) == 40
        assert m.period_at(200) == 10
        assert m.period_at(500) == 10
        # the number of periods grows by 1 / period per minute
        for t in (0, 50, 150, 300):
            rate = (m.cycles(t + 0.01) - m.cycles(t)) / 0.01
            assert rate == pytest.approx(1 / m.period_at(t), rel=1e-3)
        assert m.is_on(1)
        assert not m.is_on(25)
    flat = make("chirp", period_start_min=20, period_end_min=20, sweep_min=10)
    assert flat.cycles(50) == pytest.approx(2.5)


# ------------------------------------------------------------------ per cell
def test_all_cells_invert_and_dilate():
    seg = two_cells()
    cells = lit(make("all_cells", channel="638"), Context(seg=seg))
    assert np.array_equal(cells > 0, seg > 0)
    grown = lit(make("all_cells", channel="638", dilate_um=3), Context(seg=seg))
    assert grown[4, 10]
    assert grown[2, 10]
    assert not grown[1, 10]
    background = lit(make("all_cells", channel="638", invert=True), Context(seg=seg))
    assert np.array_equal(background > 0, seg == 0)


def test_random_cells_draws_once_per_track():
    seg = np.zeros((60, 80), np.int32)
    for i in range(40):
        r, c = divmod(i, 8)
        seg[r * 12 : r * 12 + 8, c * 10 : c * 10 + 8] = i + 1
    m = make("random_cells", channel="638", fraction=0.5, seed=3)
    assert m._requirements_list[0][3]  # it needs tracks
    first = lit(m, Context(seg=seg))
    again = lit(
        make("random_cells", channel="638", fraction=0.5, seed=3), Context(seg=seg)
    )
    assert np.array_equal(first, again)
    n_lit = len(np.unique(seg[first > 0]))
    assert 10 <= n_lit <= 30
    # a cell keeps its draw when the others are gone
    keep = int(np.unique(seg[first > 0])[0])
    only = np.where(seg == keep, seg, 0)
    assert lit(m, Context(seg=only)).max() == 1
    other = make("random_cells", channel="638", fraction=0.5, seed=4)
    assert not np.array_equal(lit(other, Context(seg=seg)), first)


def test_select_cells_by_each_property():
    seg = two_cells()
    raw = np.where(seg == 1, 100.0, np.where(seg == 2, 10.0, 0.0))

    def chosen(**kwargs):
        p = lit(make("select_cells", channel="638", **kwargs), Context(raw, seg))
        return sorted(int(i) for i in np.unique(seg[p > 0]) if i)

    assert chosen(by="area", min=300) == [2]
    assert chosen(by="area", max=300) == [1]
    assert chosen(by="intensity", min=50) == [1]
    assert chosen(by="distance_center", top_n=-1) == [2]
    assert chosen(by="distance_center", top_n=1) == [1]
    assert chosen(by="distance_edge", min=0) == [1, 2]
    m = make("select_cells", channel="638", by="intensity", min=1)
    assert any(req[1] for req in m._requirements_list)  # raw requested
    with pytest.raises(ValueError, match="all_cells"):
        known_models["select_cells"](channel="638", by="area")
    with pytest.raises(ValueError, match="by must be"):
        known_models["select_cells"](channel="638", by="perimeter", min=1)
    with pytest.raises(ValueError, match="distance_edge"):
        known_models["select_cells"](channel="638", by="area", band_um=5)


def test_select_cells_band_lights_the_colony_edge():
    seg = np.zeros((60, 80), np.int32)
    seg[10:50, 10:70] = 1
    seg[10:50, 40:70] = 2
    band = lit(
        make("select_cells", channel="638", by="distance_edge", band_um=5),
        Context(seg=seg),
    )
    assert band[12, 30]
    assert band[30, 12]
    assert band[47, 40]
    assert not band[30, 40]  # deep inside
    assert not band[5, 30]  # outside the colony


# ------------------------------------------------------------------ per nucleus
def test_nucleus_clamp_controllers():
    seg = two_cells()
    raw = np.where(seg == 1, 25.0, np.where(seg == 2, 75.0, 0.0))
    ctx = Context(raw, seg)
    bang = lit(make("nucleus_clamp", channel="545", target=50), ctx)
    assert bang[10, 10] == 1
    assert bang[40, 50] == 0
    p = lit(make("nucleus_clamp", channel="545", target=50, controller="p", kp=1), ctx)
    assert p[10, 10] == pytest.approx(0.5)
    assert p[40, 50] == 0
    pi = make("nucleus_clamp", channel="545", target=50, controller="pi", kp=0, ki=0.1)
    assert pi.use_tracks
    doses = [lit(pi, Context(raw, seg, time=60 * m))[10, 10] for m in (0, 1, 2, 3)]
    # 0.5 relative error x 0.1 per minute, summed
    assert doses == pytest.approx([0, 0.05, 0.1, 0.15])
    dithered = make(
        "nucleus_clamp", channel="545", target=50, controller="p", dose_mode="dither"
    )
    assert set(np.unique(lit(dithered, ctx))) <= {0.0, 1.0}
    with pytest.raises(ValueError, match="controller"):
        known_models["nucleus_clamp"](channel="545", controller="pid")


def test_population_clamp_actuators():
    seg = two_cells()
    raw = np.where(seg > 0, 40.0, 0.0)
    duty = make("population_clamp", channel="545", target=80, controller="p", kp=1)
    assert lit(duty, Context(raw, seg)).max() == pytest.approx(0.5)

    exposure = make(
        "population_clamp",
        channel="545",
        target=80,
        controller="p",
        actuator="exposure",
        min_value=10,
        max_value=110,
    )
    ctx = Context(raw, seg)
    assert lit(exposure, ctx).min() == 1
    assert ctx.requests == [("exposure", "stimulation", 60.0)]
    ctx = Context(raw, seg, time=60)
    lit(exposure, ctx)
    assert ctx.requests == []  # unchanged: not asked again

    prop = make(
        "population_clamp",
        channel="545",
        target=80,
        controller="bang_bang",
        actuator="property",
        device="Laser",
        property="Power",
        min_value=0,
        max_value=100,
    )
    ctx = Context(raw, seg)
    lit(prop, ctx)
    assert ctx.requests == [("property", "stimulation", "Laser", "Power", 100.0)]
    with pytest.raises(ValueError, match="min_value"):
        known_models["population_clamp"](channel="545", target=1, actuator="exposure")
    with pytest.raises(ValueError, match="device"):
        known_models["population_clamp"](
            channel="545", target=1, actuator="property", min_value=0, max_value=1
        )


def test_relative_clamp_calibrates_then_holds_a_fraction_of_each_range():
    seg = two_cells()
    m = make(
        "relative_clamp",
        channel="545",
        calibrate_min=10,
        targets=[0.5, 0.9],
        step_min=30,
    )

    def run(minute, a, b):
        raw = np.where(seg == 1, a, np.where(seg == 2, b, 0.0))
        return lit(m, Context(raw, seg, time=minute * 60))

    assert run(0, 100, 100).max() == 0  # dark phase
    assert run(9, 100, 1000).max() == 0  # cell 1 dark value 100, cell 2 1000
    assert run(10, 400, 1100).min() == 1  # lit phase
    run(19, 400, 1100)  # cell 1 lit value 400 (2 log2 steps); cell 2 too small a range
    p = run(20, 200, 1000)  # target 0.5 of [100, 400] = 250: cell 1 below, lit
    assert p[10, 10] == 1
    assert p[40, 50] == 0
    assert run(21, 300, 1000)[10, 10] == 0  # above the goal
    assert run(55, 300, 1000)[10, 10] == 1  # target 0.9 from minute 50: goal 370
    assert m.target_at(500) == 0.9  # the last target is held
    # a reporter that light dims: lit while above the goal
    m2 = make("relative_clamp", channel="545", calibrate_min=10)
    for minute, value in ((0, 400), (10, 100)):
        lit(m2, Context(np.where(seg > 0, value, 0.0), seg, time=minute * 60))
    assert lit(m2, Context(np.where(seg > 0, 300.0, 0), seg, time=1200))[10, 10] == 1
    assert lit(m2, Context(np.where(seg > 0, 200.0, 0), seg, time=1260))[10, 10] == 0
    assert len({r[0] for r in m._requirements_list}) == 1


# ------------------------------------------------------------------ adaptive
def test_wrappers_merge_their_needs_with_the_inner_methods():
    exp = make_experiment("exp.00")
    m = make(
        "recenter",
        channel="545",
        inner={"method": "nucleus_clamp", "channel": "545", "target": 5},
    )
    [req] = m.initialize(exp)
    assert req.needs_raw
    assert req.needs_seg
    assert {r[0] for r in m._requirements_list} == {"545"}
    assert check_kwargs(known_models["recenter"], {"channel": "545"}) == []
    # no inner method: the field is dark
    bare = make("recenter", channel="545")
    assert lit(bare, Context(seg=two_cells(), position=None)).max() == 0
    applied, refused = m.update(**{"inner.target": 9, "deadband_um": 3, "inner": {}})
    assert applied == {"deadband_um": 3, "inner.target": 9}
    assert "inner" in refused
    assert m.inner.target == 9


def test_camera_to_stage_uses_the_affine_and_the_binning():
    m = make("full_on", pixel_um=0.5)
    assert m.camera_to_stage(2, 4) == (2.0, 1.0)
    flipped = make("full_on", camera_to_stage=((0.0, -0.5), (0.5, 0.0)))
    assert flipped.camera_to_stage(2, 4) == (-1.0, 2.0)
    flipped.update_binning(2)
    assert flipped.camera_to_stage(2, 4) == (-2.0, 4.0)


def test_recenter_moves_the_stage_and_carries_the_pattern():
    seg = np.zeros((60, 80), np.int32)
    seg[40:50, 60:70] = 1  # centroid (44.5, 64.5); field center (29.5, 39.5)
    m = make(
        "recenter",
        channel="545",
        deadband_um=5,
        max_step_um=1000,
        inner={"method": "all_cells", "channel": "545"},
    )
    ctx = Context(seg=seg, position={"x": 100.0, "y": 200.0, "z": 0.0})
    p = lit(m, ctx)
    assert ctx.requests == [("position", 125.0, 215.0)]
    # the pattern is shown after the move: the cell will be centered
    assert p[25:35, 35:45].min() == 1
    assert p[45, 65] == 0
    # inside the deadband nothing moves
    near = make("recenter", channel="545", deadband_um=50)
    ctx = Context(seg=seg, position={"x": 0.0, "y": 0.0})
    lit(near, ctx)
    assert ctx.requests == []
    capped = make("recenter", channel="545", deadband_um=1, max_step_um=10)
    ctx = Context(seg=seg, position={"x": 0.0, "y": 0.0})
    lit(capped, ctx)
    [(_, x, y)] = ctx.requests
    assert np.hypot(x, y) == pytest.approx(10)


def blob_frame(dy=0.0, dx=0.0, shape=(64, 64)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    frame = np.zeros(shape)
    for cy, cx in ((20, 20), (40, 30), (25, 45)):
        frame += np.exp(-((yy - cy - dy) ** 2 + (xx - cx - dx) ** 2) / 8)
    return frame * 1000


@pytest.mark.parametrize("reference", ["first", "previous"])
def test_drift_correct_moves_back_by_the_drift(reference):
    m = make("drift_correct", shape=(64, 64), channel="545", reference=reference)
    pos = {"x": 0.0, "y": 0.0}
    first = blob_frame()
    ctx = Context(first, history=[first], position=pos)
    lit(m, ctx)
    assert ctx.requests == []
    drifted = blob_frame(dy=4, dx=-3)
    ctx = Context(drifted, history=[first, drifted], position=pos)
    lit(m, ctx)
    [(_, x, y)] = ctx.requests
    assert (x, y) == (pytest.approx(-3, abs=0.2), pytest.approx(4, abs=0.2))
    # after the move the content is back: no further correction
    ctx = Context(first, history=[drifted, first], position={"x": x, "y": y})
    lit(m, ctx)
    assert ctx.requests == []


def test_auto_exposure_scales_towards_the_target():
    m = make("auto_exposure", channel="545", target=0.5, max_change=0.5)
    frame = np.full((60, 80), 16383, np.uint16)  # a quarter of 16-bit range
    ctx = Context(frame)
    ctx.exposure_ms = 100
    lit(m, ctx)
    assert ctx.requests == [("exposure", "545", 150.0)]  # capped at +50 %
    assert m.next_exposure(frame, 100) == 150
    gentle = make("auto_exposure", channel="545", target=0.5, max_change=2)
    assert gentle.next_exposure(frame, 100) == pytest.approx(200, rel=1e-3)
    assert gentle.next_exposure(frame, 900) == 1000  # max_ms
    twelve_bit = make("auto_exposure", channel="545", target=0.5, full_scale=4095)
    assert twelve_bit.next_exposure(
        np.full((4, 4), 2048, np.uint16), 50
    ) == pytest.approx(50, rel=1e-3)
    on_target = Context(np.full((60, 80), 32767, np.uint16))
    on_target.exposure_ms = 100
    lit(make("auto_exposure", channel="545", target=0.5), on_target)
    assert on_target.requests == []


def test_settings_schedule_requests_each_step_once():
    m = make(
        "settings_schedule",
        setting="property",
        device="Laser",
        property="Power",
        steps=[[60, 20], [0, 10]],
    )
    asked = []
    for minute in (0, 30, 59, 60, 90):
        ctx = Context(time=minute * 60)
        lit(m, ctx)
        asked.append(ctx.requests)
    assert asked == [
        [("property", "stimulation", "Laser", "Power", 10)],
        [],
        [],
        [("property", "stimulation", "Laser", "Power", 20)],
        [],
    ]
    config = make(
        "settings_schedule",
        channel="545",
        setting="config",
        group="Filter",
        steps=[[0, "GFP"]],
    )
    ctx = Context()
    lit(config, ctx)
    assert ctx.requests == [("config", "545", "Filter", "GFP")]
    with pytest.raises(ValueError, match="group"):
        known_models["settings_schedule"](setting="config", steps=[[0, "a"]])
    with pytest.raises(ValueError, match="pair"):
        known_models["settings_schedule"](setting="exposure", steps=[5])


# ------------------------------------------------------------------ diagnostic
def test_game_of_life_reads_the_board_back_and_steps():
    from pyclm.core.patterns.diagnostic import life_step

    blinker = np.zeros((5, 5), bool)
    blinker[2, 1:4] = True
    assert np.array_equal(life_step(life_step(blinker)), blinker)
    assert life_step(blinker)[1:4, 2].all()

    m = make("game_of_life", shape=(100, 100), cells=10, fill=0.6, threshold=500)
    assert m._stim_requested
    assert m._stim_request_raw
    first = lit(m, Context(time=0))
    board = m.board.copy()
    assert np.array_equal(m.read(first * 1000), board)
    # a faithful camera: the next board is Conway's step of the projected one
    second = lit(m, Context(first * 1000, time=60))
    assert m.agreement == 1.0
    assert np.array_equal(m.board, life_step(board))
    assert np.array_equal(second, m.draw(m.board))
    # a camera shifted by half a square misreads the board
    shifted = make("game_of_life", shape=(100, 100), cells=10, fill=0.6, threshold=500)
    image = lit(shifted, Context(time=0))
    lit(shifted, Context(np.roll(image, 5, axis=1) * 1000, time=60))
    assert shifted.agreement < 0.9
    with pytest.raises(ValueError, match="threshold"):
        known_models["game_of_life"]()


# ------------------------------------------------------------------ the run
WRAPPED_TOML = """format_version = 1

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
method = "pulsed"
period_min = 30
on_min = 10

[pattern.inner]
method = "circle"
radius_um = 100
"""


def test_check_validates_the_inner_method(tmp_path):
    import shutil

    from test_check import RESOURCES, TOMLS

    from pyclm.check import check_directory

    for name in ("pyclm_config.toml", "PositionList.pos"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    for stem in ("bar10", "bar025"):
        (tmp_path / f"{stem}.toml").write_text(WRAPPED_TOML)
    report = check_directory(tmp_path)
    assert report.ok, report.text()

    (tmp_path / "bar10.toml").write_text(WRAPPED_TOML.replace("radius_um", "radius"))
    report = check_directory(tmp_path)
    assert not report.ok
    assert "[pattern.inner]" in report.text()


def test_the_pattern_process_builds_the_inner_method():
    from pyclm.core.pattern_process import PatternProcess
    from pyclm.core.patterns.composite import WrappingPattern
    from pyclm.core.queues import AllQueues

    aq = AllQueues()
    pp = PatternProcess(aq)
    exp = make_experiment(
        "exp.00",
        pattern_method="pulsed",
        pattern_kwargs={
            "period_min": 10,
            "on_min": 5,
            "inner": {"method": "circle", "radius_um": 5},
        },
    )
    pp.request_method(exp)
    pp.initialize(CameraProperties(ROI(0, 0, 40, 20), 1.0, ((0.0, 1.0), (1.0, 0.0))))
    pp.initialize_models()
    model = pp.models["exp.00"]
    assert isinstance(model, WrappingPattern)
    assert model.inner.pattern_shape == (20, 40)
    assert model.inner.camera_properties.camera_to_stage == ((0.0, 1.0), (1.0, 0.0))


def test_events_snapshot_the_stage_position():
    from pyclm.core.events import AcquisitionEvent
    from pyclm.core.experiments import MicroscopePosition

    pos = MicroscopePosition(1.0, 2.0, 3.0)
    event = AcquisitionEvent("exp.00", pos, None)
    pos.x = 50.0
    assert event.stage_xy == (1.0, 2.0)
    assert AcquisitionEvent("exp.00", None, None).stage_xy is None


def test_the_tracker_follows_a_stage_move():
    from test_tracking import blobs

    from pyclm.core.datatypes import SegmentationData
    from pyclm.core.events import AcquisitionEvent
    from pyclm.core.experiments import MicroscopePosition, TrackingConfig
    from pyclm.core.tracking_process import TrackingProcess

    exp = make_experiment("exp.00")
    exp.tracking = TrackingConfig("centroid", max_distance_um=5.0)

    def frame(t, x, centres):
        event = AcquisitionEvent(
            "exp.00",
            MicroscopePosition(x, 0.0, 0.0),
            exp.channels["545"].channel_id,
            index={"t": t, "c": "545"},
        )
        event.pixel_width_um = 2.0
        return SegmentationData(event, blobs(centres))

    def run(camera_to_stage):
        proc = TrackingProcess()
        proc.camera_to_stage = camera_to_stage
        proc.request_method(exp, "545")
        out = []
        proc.publish = out.append
        proc.handle_data(frame(0, 0.0, [(20, 20), (20, 44)]))
        # the stage moved 24 um in x: the content moved 12 px to the left
        proc.handle_data(frame(1, 24.0, [(20, 8), (20, 32)]))
        return [r.track_id for r in sorted(out[-1].rows, key=lambda r: r.x)]

    assert run(None) == [1, 2]
    assert run(((2.0, 0.0), (0.0, 2.0))) == [1, 2]
    # the conversion: plain, and through a camera turned by 90 degrees
    proc = TrackingProcess()
    proc.request_method(exp, "545")
    assert proc.stage_to_pixels(24.0, 0.0, 2.0) == (0.0, 12.0)
    proc.camera_to_stage = ((0.0, 2.0), (2.0, 0.0))
    assert proc.stage_to_pixels(24.0, 0.0, 2.0) == (12.0, 0.0)


def test_a_position_change_moves_every_grid_tile():
    import threading

    from helpers import make_plan, make_schedule

    from pyclm.core import settings
    from pyclm.core.experiments import MicroscopePosition
    from pyclm.core.manager import Manager
    from pyclm.core.messages import SettingsRequestMessage
    from pyclm.core.queues import AllQueues
    from pyclm.core.storage.events import EventLog

    exp = make_experiment("exp.00")
    manager = Manager(AllQueues(), threading.Event())
    manager.initialize(
        make_plan(make_schedule([exp], steps=3, interval=0.02)),
        event_log=EventLog(None),
        status_path=None,
        health=lambda: {},
    )
    grid = manager.positions["exp.00"]
    grid.tiles = [MicroscopePosition(x, y, 5.0) for x, y in ((0, 0), (100, 0))]
    grid.x, grid.y = 50.0, 0.0
    manager.apply_settings(
        SettingsRequestMessage(
            "exp.00",
            0,
            [
                settings.position("x", 60.0),
                settings.position("y", -5.0),
                settings.position("pfs_offset", 7.0),
            ],
        )
    )
    assert [(t.x, t.y) for t in grid.tiles] == [(10.0, -5.0), (110.0, -5.0)]
    assert all(t.extras["PFSOffset"] == 7.0 for t in grid.tiles)


def test_the_controller_reads_the_pixel_size_affine():
    from pyclm.controller import camera_to_stage_matrix

    class Core:
        def __init__(self, affine):
            self.affine = affine

        def getPixelSizeAffine(self):
            return self.affine

    assert camera_to_stage_matrix(Core((0.5, 0, 0, 0, -0.5, 0))) == (
        (0.5, 0.0),
        (0.0, -0.5),
    )
    assert camera_to_stage_matrix(Core((0, 0, 0, 0, 0, 0))) is None
    assert camera_to_stage_matrix(object()) is None
