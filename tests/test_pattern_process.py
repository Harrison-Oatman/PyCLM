"""
Unit tests for PatternProcess: dock keying by absolute timepoint, graceful
handling of unrequested data, per-instance method registration, and the
subscriptions it declares to the router.
"""

import logging

import numpy as np
import pytest
from helpers import make_experiment, make_plan, make_schedule

from pyclm.core.datatypes import AcquisitionData, CameraPattern
from pyclm.core.events import AcquisitionEvent
from pyclm.core.experiments import MicroscopePosition
from pyclm.core.pattern_process import PatternProcess, RequestPattern
from pyclm.core.patterns import (
    ROI,
    CameraProperties,
    PatternContext,
    PatternMethod,
    known_models,
)
from pyclm.core.queues import AllQueues
from pyclm.core.router import Subscription


class StubPattern(PatternMethod):
    name = "stub"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_requirement("545", raw=True, seg=False)
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        h, w = self.pattern_shape
        return np.zeros((int(h), int(w)), dtype=np.float32)


def make_process(exp):
    aq = AllQueues()
    pp = PatternProcess(aq)
    pp.register_method(StubPattern, "stub")
    pp.initialize(CameraProperties(ROI(0, 0, 64, 48), 1.0))
    requirements = pp.request_method(exp)
    pp.initialize_models()
    return aq, pp, requirements


def make_raw(exp, t_index):
    event = AcquisitionEvent(
        exp.experiment_name,
        MicroscopePosition(0.0, 0.0, 0.0, label=exp.experiment_name),
        exp.channels["545"].channel_id,
        index={"t": t_index, "p": exp.experiment_name, "c": "545"},
    )
    return AcquisitionData(event, np.zeros((48, 64), dtype=np.uint16))


def test_registration_is_per_instance():
    exp = make_experiment("exp.00", pattern_method="stub")
    _, pp, _ = make_process(exp)

    assert "stub" in pp.known_models
    assert "stub" not in known_models
    assert "stub" not in PatternProcess(AllQueues()).known_models


def test_data_with_matching_absolute_index_triggers_generate():
    exp = make_experiment("exp.00", pattern_method="stub", t_delay=2)
    aq, pp, requirements = make_process(exp)

    # the manager sends the absolute timepoint in both the request and the event
    pp.handle_message(RequestPattern(5, 42.0, "exp.00", requirements))
    assert ("exp.00", 5) in pp.docks

    pp.handle_data(make_raw(exp, t_index=5))

    out = aq.pattern_to_slm.get_nowait()
    assert isinstance(out, CameraPattern)
    assert out.experiment == "exp.00"
    assert out.data.shape == (48, 64)

    model = pp.models["exp.00"]
    assert len(model.contexts) == 1
    assert model.contexts[0].time == 42.0
    assert ("exp.00", 5) not in pp.docks


def test_data_without_request_is_dropped_with_warning(caplog):
    exp = make_experiment("exp.00", pattern_method="stub")
    aq, pp, _ = make_process(exp)

    with caplog.at_level(logging.WARNING):
        pp.handle_data(make_raw(exp, t_index=3))

    assert aq.pattern_to_slm.empty()
    assert "no pattern was requested" in caplog.text
    assert pp.models["exp.00"].contexts == []


def test_subscriptions_follow_the_requirements_at_pattern_cadence():
    exp = make_experiment("exp.00", pattern_method="stub")
    _, pp, requirements = make_process(exp)
    plan = make_plan(make_schedule([exp], steps=2), {"exp.00": requirements})

    assert pp.subscriptions(plan) == [
        Subscription("pattern", "exp.00", "545", "raw", "pattern")
    ]


def test_unknown_method_is_rejected_at_request_time():
    exp = make_experiment("exp.00", pattern_method="does_not_exist")
    pp = PatternProcess(AllQueues())

    with pytest.raises(AssertionError, match="not a registered method"):
        pp.request_method(exp)


# ------------------------------------------------------------------ history
class HistoryPattern(PatternMethod):
    name = "history_stub"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_requirement("545", raw=True, history=3)
        self.seen = []

    def generate(self, context):
        self.seen.append(
            (
                context.t,
                [int(a.max()) for a in context.history("545", kind="raw")],
                None
                if context.last_pattern() is None
                else float(context.last_pattern().max()),
                context.generation,
            )
        )
        h, w = self.pattern_shape
        return np.full((int(h), int(w)), float(context.t), dtype=np.float32)


def make_raw_value(exp, t_index, value):
    data = make_raw(exp, t_index)
    data.data = np.full((48, 64), value, dtype=np.uint16)
    return data


def test_context_history_keeps_the_last_n_deliveries_and_previous_patterns():
    exp = make_experiment("exp.00", pattern_method="history_stub")
    aq = AllQueues()
    pp = PatternProcess(aq)
    pp.register_method(HistoryPattern, "history_stub")
    pp.initialize(CameraProperties(ROI(0, 0, 64, 48), 1.0))
    requirements = pp.request_method(exp)
    pp.initialize_models()
    assert requirements[0].history == 3

    for t in range(5):
        pp.handle_message(RequestPattern(t, float(t), "exp.00", requirements))
        pp.handle_data(make_raw_value(exp, t, 10 + t))

    model = pp.models["exp.00"]
    assert [s[0] for s in model.seen] == [0, 1, 2, 3, 4]
    # oldest first, current last, never more than the requested depth
    assert [s[1] for s in model.seen] == [
        [10],
        [10, 11],
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 14],
    ]
    # the previous pattern is what generate() returned last time
    assert [s[2] for s in model.seen] == [None, 0.0, 1.0, 2.0, 3.0]
    assert [s[3] for s in model.seen] == [0, 1, 2, 3, 4]
    assert len(pp.states["exp.00"].patterns) == 2  # PatternMethod.pattern_history
    assert pp.docks == {}


def test_context_rejects_unrequested_history():
    exp = make_experiment("exp.00", pattern_method="stub")
    _, pp, requirements = make_process(exp)
    pp.handle_message(RequestPattern(0, 0.0, "exp.00", requirements))
    pp.handle_data(make_raw(exp, 0))
    context = PatternContext(pp.states["exp.00"], exp)

    assert len(context.history("545", kind="raw")) == 1
    with pytest.raises(ValueError, match="not requested"):
        context.history("545", kind="seg")
    with pytest.raises(ValueError, match="not requested"):
        context.tracks("545")
