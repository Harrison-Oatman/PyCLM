"""
Unit tests for the Manager's event emission: which events are produced at
which timepoints, that pattern requests and acquisition events agree on the
timepoint index (regression for the t_delay dock-key mismatch), and that the
wait loop does not spin.
"""

import threading
from time import perf_counter, process_time

import pytest
from helpers import drain, make_experiment, make_plan, make_schedule

from pyclm.core.manager import Manager
from pyclm.core.messages import (
    AcquisitionEventMessage,
    CloseMessage,
    UpdatePatternEventMessage,
    UpdatePositionEventMessage,
    UpdateZPositionMessage,
)
from pyclm.core.pattern_process import RequestPattern
from pyclm.core.patterns import AcquiredImageRequest
from pyclm.core.queues import AllQueues


def run_manager(schedule, requirements):
    aq = AllQueues()
    manager = Manager(aq, threading.Event())
    manager.initialize(make_plan(schedule, requirements))
    manager.process()
    return aq, manager


def raw_requirement(exp, channel="545"):
    return [AcquiredImageRequest(exp.channels[channel].channel_id, True, False)]


@pytest.mark.parametrize("t_delay", [0, 1, 2])
def test_pattern_request_index_matches_acquisition_index(t_delay):
    exp = make_experiment("exp.00", t_delay=t_delay)
    schedule = make_schedule([exp], steps=4)
    cid = exp.channels["545"].channel_id

    aq, _ = run_manager(schedule, {"exp.00": raw_requirement(exp)})

    requests = [
        m for m in drain(aq.manager_to_pattern) if isinstance(m, RequestPattern)
    ]
    events = [
        m.event
        for m in drain(aq.manager_to_microscope)
        if isinstance(m, AcquisitionEventMessage)
    ]
    channel_events = [e for e in events if e.channel_id == cid]

    expected_t = list(range(t_delay, 4))
    assert [r.t_index for r in requests] == expected_t
    assert [e.t_index for e in channel_events] == expected_t
    # every requested timepoint routes its raw frame to the pattern process
    assert all(e.index["c"] == "545" for e in channel_events)


def test_pattern_cadence_follows_lcm_of_required_channels():
    exp = make_experiment("exp.00", every_t=2)
    schedule = make_schedule([exp], steps=5)
    cid = exp.channels["545"].channel_id

    aq, manager = run_manager(schedule, {"exp.00": raw_requirement(exp)})

    assert manager.plan.pattern_lcm("exp.00") == 2
    requests = [
        m for m in drain(aq.manager_to_pattern) if isinstance(m, RequestPattern)
    ]
    assert [r.t_index for r in requests] == [0, 2, 4]

    events = [
        m.event
        for m in drain(aq.manager_to_microscope)
        if isinstance(m, AcquisitionEventMessage)
    ]
    assert [e.t_index for e in events if e.channel_id == cid] == [0, 2, 4]
    # stimulation runs every timepoint regardless
    assert [e.t_index for e in events if e.needs_slm] == [0, 1, 2, 3, 4]


def test_t_stop_ends_experiment_early():
    exp = make_experiment("exp.00", t_delay=1, t_stop=2)
    schedule = make_schedule([exp], steps=6)

    aq, _ = run_manager(schedule, {"exp.00": raw_requirement(exp)})

    events = [
        m.event
        for m in drain(aq.manager_to_microscope)
        if isinstance(m, AcquisitionEventMessage)
    ]
    assert sorted({e.t_index for e in events}) == [1, 2]


def test_event_order_within_timepoint():
    exp = make_experiment("exp.00")
    schedule = make_schedule([exp], steps=1)

    aq, _ = run_manager(schedule, {"exp.00": raw_requirement(exp)})

    messages = drain(aq.manager_to_microscope)
    kinds = [type(m) for m in messages]
    assert kinds == [
        UpdatePositionEventMessage,
        UpdatePatternEventMessage,
        AcquisitionEventMessage,  # stimulation (needs_slm)
        AcquisitionEventMessage,  # imaging channel
        CloseMessage,
    ]
    assert messages[2].event.needs_slm
    assert not messages[3].event.needs_slm
    # the SLM buffer sees the same update-pattern event as the microscope
    slm_messages = drain(aq.manager_to_slm_buffer)
    assert slm_messages[0].event.id == messages[1].event.id


def test_close_is_sent_to_every_addressed_process():
    exp = make_experiment("exp.00")
    aq, _ = run_manager(make_schedule([exp], steps=1), {"exp.00": []})

    # the writer and segmentation exit on the router's stream close instead
    for queue in (
        aq.manager_to_microscope,
        aq.manager_to_slm_buffer,
        aq.manager_to_pattern,
    ):
        assert isinstance(drain(queue)[-1], CloseMessage)


def test_z_update_from_microscope_is_applied():
    exp = make_experiment("exp.00")
    schedule = make_schedule([exp], steps=1)
    aq = AllQueues()
    manager = Manager(aq, threading.Event())
    manager.initialize(make_plan(schedule))

    aq.microscope_to_manager.put(UpdateZPositionMessage(12.5, "exp.00"))

    assert manager.drain_inboxes() is True
    assert manager.positions["exp.00"].z == 12.5
    assert manager.drain_inboxes() is False


def test_wait_loop_sleeps_instead_of_spinning():
    exp = make_experiment("exp.00")
    schedule = make_schedule([exp], steps=2, interval=0.3)

    cpu0, wall0 = process_time(), perf_counter()
    run_manager(schedule, {"exp.00": []})
    cpu, wall = process_time() - cpu0, perf_counter() - wall0

    assert wall >= 0.25
    # a spinning loop would burn roughly one core for the whole wait
    assert cpu < 0.5 * wall


def test_stop_event_interrupts_wait():
    exp = make_experiment("exp.00")
    schedule = make_schedule([exp], steps=3, interval=5.0)
    aq = AllQueues()
    stop = threading.Event()
    manager = Manager(aq, stop)
    manager.initialize(make_plan(schedule))

    thread = threading.Thread(target=manager.process)
    start = perf_counter()
    thread.start()
    stop.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert perf_counter() - start < 2.0
