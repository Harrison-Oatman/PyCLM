"""
Unit tests for PatternProcess: dock keying by absolute timepoint, graceful
handling of unrequested data, and per-instance method registration.
"""

import logging

import numpy as np
import pytest
from helpers import make_experiment

from pyclm.core.datatypes import AcquisitionData, CameraPattern
from pyclm.core.events import AcquisitionEvent
from pyclm.core.experiments import MicroscopePosition
from pyclm.core.pattern_process import PatternProcess, RequestPattern
from pyclm.core.patterns import ROI, CameraProperties, PatternMethod, known_models
from pyclm.core.queues import AllQueues


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
        raw_goes_to_pattern=True,
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

    pp.handle_from_raw(make_raw(exp, t_index=5))

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
        should_stop = pp.handle_from_raw(make_raw(exp, t_index=3))

    assert should_stop is False
    assert aq.pattern_to_slm.empty()
    assert "no pattern was requested" in caplog.text
    assert pp.models["exp.00"].contexts == []


def test_unknown_method_is_rejected_at_request_time():
    exp = make_experiment("exp.00", pattern_method="does_not_exist")
    pp = PatternProcess(AllQueues())

    with pytest.raises(AssertionError, match="not a registered method"):
        pp.request_method(exp)
