"""
AcquisitionPlan: enumeration against the scheduling rules PyCLM has always
used, event order and timing offsets, routing flags, YAML round trip, HDF5
paths from the index, expected datasets, z-readiness, and the timing budget.
"""

import itertools
from math import lcm

import pytest
import useq
from helpers import make_experiment, make_plan, make_schedule

from pyclm.core.events import rel_path, storage_group
from pyclm.core.experiments import (
    ConfigGroup,
    Experiment,
    ImagingConfig,
    PatternConfig,
    SegmentationConfig,
)
from pyclm.core.patterns import AcquiredImageRequest
from pyclm.core.plan import PLAN_FILENAME, AcquisitionPlan


def raw_requirement(exp, channel="545"):
    return {
        exp.experiment_name: [
            AcquiredImageRequest(exp.channels[channel].channel_id, True, False)
        ]
    }


def reference(exp, steps, required):
    """The scheduling rules written out plainly, independent of the plan."""
    frames, requests = [], []
    stim = exp.stimulation
    for t in range(steps):
        this_t = t - exp.t_delay
        if this_t < 0 or (exp.t_stop > 0 and this_t >= exp.t_stop):
            continue
        cadence = [exp.channels[c].every_t for c in required]
        if this_t % lcm(exp.pattern.every_t, *cadence) == 0:
            requests.append(t)
        if stim.exposure > 0 and this_t % stim.every_t == 0:
            frames.append((t, "stim"))
        for name, cfg in exp.channels.items():
            if this_t % cfg.every_t == 0:
                frames.append((t, name))
    return frames, requests


def enumerate_plan(plan):
    frames, requests = [], []
    for t in range(plan.timepoints):
        for ev in plan.events_at(t):
            if ev.kind == "acquire":
                frames.append((t, "stim" if ev.is_stim else ev.channel))
            elif ev.kind == "request_pattern":
                requests.append(t)
    return frames, requests


@pytest.mark.parametrize(
    ("every_t", "t_delay", "t_stop", "stim_every"),
    list(itertools.product([1, 2, 3], [0, 1, 2], [0, 2, 5], [1, 2])),
)
def test_enumeration_matches_scheduling_rules(every_t, t_delay, t_stop, stim_every):
    exp = make_experiment(
        "exp.00",
        every_t=every_t,
        t_delay=t_delay,
        t_stop=t_stop,
        stim_every_t=stim_every,
    )
    plan = make_plan(make_schedule([exp], steps=8), raw_requirement(exp))

    assert enumerate_plan(plan) == reference(exp, 8, ["545"])

    datasets = [
        (d.t, "stim" if d.is_stim else d.channel) for d in plan.expected_datasets()
    ]
    assert datasets == reference(exp, 8, ["545"])[0]

    for t in range(8):
        assert plan.is_scheduled("exp.00", "545", t) == ((t, "545") in datasets)
        assert plan.is_scheduled("exp.00", "DMD", t) == ((t, "stim") in datasets)


def test_event_order_indices_and_offsets():
    a = make_experiment("a.00")
    b = make_experiment("b.00")
    plan = make_plan(
        make_schedule([a, b], steps=2, interval=10.0, between=0.5),
        raw_requirement(a) | raw_requirement(b),
    )

    events = plan.events_at(1)
    assert [(e.kind, e.experiment) for e in events] == [
        ("request_pattern", "a.00"),
        ("position", "a.00"),
        ("update_pattern", "a.00"),
        ("acquire", "a.00"),
        ("acquire", "a.00"),
        ("request_pattern", "b.00"),
        ("position", "b.00"),
        ("update_pattern", "b.00"),
        ("acquire", "b.00"),
        ("acquire", "b.00"),
    ]
    stim, channel = events[3], events[4]
    assert stim.is_stim
    assert stim.channel == "DMD"
    assert stim.index == {"t": 1, "p": "a.00", "c": "DMD"}
    assert channel.index == {"t": 1, "p": "a.00", "c": "545"}
    assert all(e.scheduled_offset_s == 10.0 for e in events[:5])
    assert all(e.scheduled_offset_s == 10.5 for e in events[5:])
    assert plan.time_offset_s(1) == 10.0
    assert plan.interval_s == 10.0


def test_no_position_event_when_nothing_is_acquired():
    exp = make_experiment("exp.00", every_t=2, stim_exposure=0)
    plan = make_plan(make_schedule([exp], steps=4))

    assert plan.events_at(1) == []
    assert [e.kind for e in plan.events_at(2)] == [
        "request_pattern",
        "position",
        "acquire",
    ]
    assert plan.stim_channel("exp.00") is None
    assert plan.channels("exp.00") == ["545"]


def test_pattern_requirements_and_cadence_follow_requirements():
    exp = make_experiment("exp.00", pattern_kwargs={"every_t": 2})
    reqs = {
        "exp.00": [AcquiredImageRequest(exp.channels["545"].channel_id, True, True)]
    }
    plan = make_plan(make_schedule([exp], steps=3), reqs)

    # the plan records what the pattern needs; the router turns it into deliveries
    assert plan.pattern_requirements("exp.00") == {
        "545": {"raw": True, "seg": True, "tracks": False}
    }
    assert plan.pattern_lcm("exp.00") == 2
    assert [plan.pattern_due("exp.00", t) for t in range(3)] == [True, False, True]

    # events carry identity only: nothing about consumers
    acquire = next(e for e in plan.events_at(0) if e.kind == "acquire")
    assert not hasattr(acquire, "segment")
    assert not hasattr(acquire, "raw_to_pattern")

    # t=1: no pattern due (pattern every 2)
    assert "request_pattern" not in [e.kind for e in plan.events_at(1)]

    # a plan read back from YAML resolves the same requirements from metadata
    reread = AcquisitionPlan(plan.sequence, plan.schedule)
    assert reread.pattern_requirements("exp.00") == plan.pattern_requirements("exp.00")


def test_yaml_round_trip(tmp_path):
    exp = make_experiment("exp.00", every_t=2, t_delay=1)
    schedule = make_schedule([exp], steps=5, interval=2.0, between=0.25)
    plan = make_plan(schedule, raw_requirement(exp))

    path = plan.to_yaml(tmp_path / PLAN_FILENAME)
    text = path.read_text()
    assert "pyclm" in text
    assert "useq_version" in text

    loaded = AcquisitionPlan.from_yaml(path, schedule, raw_requirement(exp))
    assert loaded.timepoints == plan.timepoints
    for t in range(plan.timepoints):
        assert loaded.events_at(t) == plan.events_at(t)

    # the YAML alone carries the cadence: no requirements needed to re-derive it
    loaded_bare = AcquisitionPlan.from_yaml(path, schedule)
    assert loaded_bare.pattern_lcm("exp.00") == plan.pattern_lcm("exp.00")


def test_hdf5_paths_from_index():
    assert storage_group("545", False) == "channel_545"
    assert storage_group("DMD", True) == "stim_aq"
    assert rel_path({"t": 12, "p": "x", "c": "545"}, False) == "00012/channel_545/"
    assert rel_path({"t": 12, "p": "x", "c": "DMD"}, True) == "00012/stim_aq/"


def test_stimulation_channel_name_from_channel_group():
    exp = make_experiment("exp.00")
    plan = make_plan(make_schedule([exp]))
    assert plan.stim_channel("exp.00") == "DMD"
    assert plan.channels("exp.00") == ["DMD", "545"]
    assert plan.imaging_config("exp.00", "DMD") is exp.stimulation

    # a stimulation config with no preset in the channel group gets the fallback name
    channel = ImagingConfig("exp.01", config_groups=[ConfigGroup("Channel", "545")])
    stim = ImagingConfig("exp.01", config_groups=[ConfigGroup("LightPath", "DMD")])
    bare = Experiment(
        "exp.01",
        {"545": channel},
        stim,
        SegmentationConfig("none"),
        PatternConfig("full_on"),
    )
    plan = make_plan(make_schedule([bare]))
    assert plan.stim_channel("exp.01") == "stimulation"
    assert plan.channels("exp.01") == ["stimulation", "545"]


def test_pfs_offset_recorded_but_not_executed():
    exp = make_experiment("exp.00")
    schedule = make_schedule([exp])
    schedule.positions["exp.00"].extras["PFSOffset"] = 9969.0
    plan = make_plan(schedule)

    af = plan.sequence.stage_positions[0].sequence.autofocus_plan
    assert af is not None
    assert af.autofocus_motor_offset == 9969.0
    assert {e.kind for t in range(plan.timepoints) for e in plan.events_at(t)} <= {
        "request_pattern",
        "position",
        "update_pattern",
        "acquire",
    }


def test_z_plan_enumerates_imaging_channels_only():
    exp = make_experiment("exp.00")
    schedule = make_schedule([exp])
    plan = make_plan(schedule)

    pos = plan.sequence.stage_positions[0]
    with_z = useq.Position(
        x=pos.x,
        y=pos.y,
        z=pos.z,
        name=pos.name,
        sequence=pos.sequence.replace(z_plan={"range": 4.0, "step": 2.0}),
    )
    zplan = AcquisitionPlan(plan.sequence.replace(stage_positions=(with_z,)), schedule)

    structure = zplan.structure("exp.00")
    assert [e.channel.config for e in structure] == ["DMD", "545", "545", "545"]
    assert [e.index.get("z") for e in structure] == [1, 0, 1, 2]


def test_over_budget_flags_slow_timepoints():
    exp = make_experiment("exp.00", exposure=20, stim_exposure=20)
    plan = make_plan(make_schedule([exp], steps=3, interval=1.0))

    assert plan.estimate_timepoint_s(0, settle_s=1.0) == pytest.approx(2.04)
    assert [t for t, _ in plan.over_budget(settle_s=1.0)] == [0, 1, 2]
    assert plan.over_budget(settle_s=0.1) == []


def test_plan_rejects_mismatched_schedule():
    schedule = make_schedule([make_experiment("exp.00")])
    other = make_schedule([make_experiment("other.00")])
    plan = make_plan(schedule)

    with pytest.raises(ValueError, match="do not match"):
        AcquisitionPlan(plan.sequence, other)
