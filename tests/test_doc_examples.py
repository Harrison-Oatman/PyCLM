"""
The pattern methods shown in the user docs (documentation/examples/) run
against synthetic data: the tracking-based leader cells and intensity
programme, the KTR clamp on two named segmentations, and the
``tracks = true`` switch on the per-cell base classes.
"""

import importlib.util
from pathlib import Path

import numpy as np
from helpers import make_experiment

from pyclm.core.datatypes import AcquisitionData, SegmentationData, TrackingData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.experiments import SegmentationConfig
from pyclm.core.patterns import ROI, CameraProperties, DataDock, PatternContext
from pyclm.core.patterns.cell_intensity_patterns import BinaryNucleusClampModel
from pyclm.core.patterns.fbc_cell_movement import RotateCcwModel
from pyclm.core.tracking import TrackRow

EXAMPLES = Path(__file__).parent.parent / "documentation" / "examples"
SHAPE = (64, 64)


def load_example(stem):
    spec = importlib.util.spec_from_file_location(stem, EXAMPLES / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blobs(cells, shape=SHAPE, radius=4):
    """Label image with one disc per (id, y, x)."""
    img = np.zeros(shape, dtype=np.uint32)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for label, y, x in cells:
        img[(yy - y) ** 2 + (xx - x) ** 2 <= radius**2] = label
    return img


def setup(method_cls, exp, **kwargs):
    method = method_cls(**kwargs)
    requirements = method.initialize(exp)
    method.configure_system(
        exp.experiment_name, CameraProperties(ROI(0, 0, SHAPE[1], SHAPE[0]), 1.0), exp
    )
    return method, requirements


def event_for(exp, t):
    return AcquisitionEvent(
        "exp.00",
        None,
        exp.channels["545"].channel_id,
        index={"t": t, "p": "exp.00", "c": "545"},
    )


def context_for(exp, requirements, t, time_s, cells, intensities=None):
    """A PatternContext holding tracks (and a raw frame) for one timepoint."""
    labels = blobs(cells)
    rows = [
        TrackRow(tid, tid, float(y), float(x), int((labels == tid).sum()))
        for tid, y, x in cells
    ]
    event = event_for(exp, t)
    dock = DataDock(time_s, requirements)
    if any(r.needs_raw for r in requirements):
        raw = np.full(SHAPE, 100, dtype=np.uint16)
        for tid, value in (intensities or {}).items():
            raw[labels == tid] = value
        dock.add(AcquisitionData(event, raw))
    dock.add(TrackingData(event, labels, rows))
    assert dock.check_complete()
    return PatternContext(dock, exp, t=t), labels


# ------------------------------------------------------------- leader cells
def test_leader_cells_are_chosen_once_and_lit_half():
    examples = load_example("tracking_patterns")
    exp = make_experiment("exp.00", pattern_method="leader_cells")
    method, reqs = setup(examples.LeaderCells, exp, channel="545", fraction=0.5, seed=1)
    assert reqs[0].needs_tracks
    assert not reqs[0].needs_seg

    cells = [(1, 10, 10), (2, 10, 40), (3, 40, 10), (4, 40, 40)]
    context, labels = context_for(exp, reqs, 0, 0.0, cells)
    pattern = method.generate(context)

    assert len(method.leaders) == 2
    leaders, others = method.leaders, {1, 2, 3, 4} - method.leaders
    for tid in others:
        assert pattern[labels == tid].sum() == 0
    for tid in leaders:
        lit = (pattern[labels == tid] > 0).mean()
        assert 0.3 < lit < 0.7  # the half of the cell facing counter-clockwise
    assert pattern[labels == 0].sum() == 0

    # later timepoints: the same leaders, a cell that moved keeps its light,
    # a newcomer (id 5) is not a leader, a lost leader is simply absent
    moved = [(tid, y + 2, x) for tid, y, x in cells if tid != min(leaders)] + [
        (5, 25, 25)
    ]
    context, labels = context_for(exp, reqs, 5, 300.0, moved)
    pattern = method.generate(context)
    assert method.leaders == leaders
    assert pattern[labels == 5].sum() == 0
    for tid in leaders - {min(leaders)}:
        assert pattern[labels == tid].sum() > 0


def test_leader_choice_waits_for_the_first_cells():
    examples = load_example("tracking_patterns")
    exp = make_experiment("exp.00", pattern_method="leader_cells")
    method, reqs = setup(examples.LeaderCells, exp, channel="545", fraction=0.3)

    context, _ = context_for(exp, reqs, 0, 0.0, [])
    assert method.generate(context).sum() == 0
    assert method.leaders is None

    context, _ = context_for(
        exp, reqs, 1, 60.0, [(7, 20, 20), (8, 40, 40), (9, 20, 40)]
    )
    method.generate(context)
    assert len(method.leaders) == 1
    assert method.leaders < {7, 8, 9}


# -------------------------------------------------------- intensity program
def test_intensity_program_phases_and_per_cell_targets():
    examples = load_example("tracking_patterns")
    exp = make_experiment("exp.00", pattern_method="intensity_program")
    method, reqs = setup(
        examples.IntensityProgram, exp, channel="545", dark_min=1, light_min=1, gain=2.0
    )
    assert reqs[0].needs_raw
    assert reqs[0].needs_tracks
    cells = [(1, 16, 16), (2, 48, 48)]

    # dark phase: nothing lit, the settled dark level is remembered
    context, labels = context_for(exp, reqs, 0, 0.0, cells, {1: 100, 2: 150})
    assert method.generate(context).sum() == 0
    context, labels = context_for(exp, reqs, 1, 30.0, cells, {1: 110, 2: 160})
    assert method.generate(context).sum() == 0
    assert method.low.as_dict() == {1: 110.0, 2: 160.0}

    # light phase: every cell fully lit, the lit level is remembered
    context, labels = context_for(exp, reqs, 2, 60.0, cells, {1: 300, 2: 360})
    pattern = method.generate(context)
    assert np.all(pattern[labels > 0] == 1.0)
    assert pattern[labels == 0].sum() == 0
    assert method.high.as_dict() == {1: 300.0, 2: 360.0}

    # feedback phase: targets are the midpoints (205 and 260)
    context, labels = context_for(
        exp, reqs, 4, 120.0, [*cells, (3, 16, 48)], {1: 205, 2: 100, 3: 400}
    )
    pattern = method.generate(context)
    assert np.allclose(pattern[labels == 1], 0.5)  # at target: half duty
    assert np.allclose(pattern[labels == 2], 1.0)  # far below: full light
    # a newcomer uses the population medians: target 232.5, so 400 is far above
    assert np.allclose(pattern[labels == 3], 0.0)
    assert pattern[labels == 0].sum() == 0


# ---------------------------------------------------------------- KTR clamp
def test_ktr_clamp_lights_cells_by_their_ratio():
    examples = load_example("ktr_patterns")
    exp = make_experiment("exp.00", pattern_method="ktr_clamp")
    exp.segmentations["nuclei"] = SegmentationConfig("cellpose", model="nuclei")
    exp.segmentations["cells"] = SegmentationConfig("cellpose", model="cyto3")
    method, reqs = setup(examples.KTRClamp, exp, channel="545", target=1.0, gain=2.0)
    assert reqs[0].kinds == ("raw", "seg:nuclei", "seg:cells")

    cells = blobs([(1, 16, 16), (2, 48, 48), (3, 16, 48)], radius=8)
    nuclei = blobs([(11, 16, 16), (12, 48, 48), (13, 16, 48), (14, 48, 8)], radius=3)
    image = np.full(SHAPE, 1.0, dtype=np.float32)
    image[cells > 0] = 10.0  # cytosol of every cell
    image[nuclei == 11] = 30.0  # ratio 3: far above target, full light
    image[nuclei == 12] = 5.0  # ratio 0.5: below target, no light
    image[nuclei == 13] = 10.0  # ratio 1: at target, half duty

    event = event_for(exp, 0)
    dock = DataDock(0.0, reqs)
    dock.add(AcquisitionData(event, image))
    dock.add(SegmentationData(event, nuclei, "nuclei"))
    dock.add(SegmentationData(event, cells, "cells"))
    assert dock.check_complete()
    pattern = method.generate(PatternContext(dock, exp, t=0))

    assert pattern.shape == SHAPE
    assert np.all(pattern[cells == 1] == 1.0)
    assert np.all(pattern[cells == 2] == 0.0)
    assert np.allclose(pattern[cells == 3], 0.5)
    assert pattern[cells == 0].sum() == 0  # the orphan nucleus 14 lights nothing


# ------------------------------------------------ tracks switch on the bases
def test_per_cell_bases_accept_tracks_switch():
    exp = make_experiment("exp.00")
    rotate, reqs = setup(RotateCcwModel, exp, channel="545", tracks=True)
    assert [(r.needs_seg, r.needs_tracks) for r in reqs] == [(False, True)]
    context, labels = context_for(exp, reqs, 0, 0.0, [(11, 20, 20), (12, 44, 44)])
    pattern = rotate.generate(context)
    assert pattern[labels == 0].sum() == 0
    assert 0 < (pattern[labels == 11] > 0).mean() < 1

    clamp, reqs = setup(
        BinaryNucleusClampModel, exp, channel="545", tracks=True, clamp_target=200
    )
    assert [(r.needs_raw, r.needs_seg, r.needs_tracks) for r in reqs] == [
        (True, False, True)
    ]
    context, labels = context_for(
        exp, reqs, 0, 0.0, [(11, 20, 20), (12, 44, 44)], {11: 50, 12: 500}
    )
    pattern = clamp.generate(context)
    assert np.all(pattern[labels == 11] == 1)
    assert np.all(pattern[labels == 12] == 0)

    _plain, reqs = setup(RotateCcwModel, exp, channel="545")
    assert [(r.needs_seg, r.needs_tracks) for r in reqs] == [(True, False)]
