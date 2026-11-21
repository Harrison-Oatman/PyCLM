"""
The pattern of a stimulation is chosen when the microscope reaches it, not
when the Manager schedules it: the microscope asks the SLM buffer, which
answers with the newest pattern made for a timepoint at or before the one
being run. A microscope running behind its schedule then lights cells with
their newest outlines, and a schedule-only pattern is still the one made for
its own timepoint.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
from helpers import FakeImageSource

from pyclm.core.datatypes import CameraPattern
from pyclm.core.events import UpdatePatternEvent
from pyclm.core.manager import SLMBuffer
from pyclm.core.messages import UpdatePatternEventMessage
from pyclm.core.microscope import MicroscopeProcess
from pyclm.core.queues import AllQueues
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore

AFFINE = [[0.5, 0.0, 100.0], [0.0, 0.5, 200.0]]


def make_buffer(stop_event=None):
    aq = AllQueues()
    slm = SLMBuffer(aq, stop_event)
    slm.initialize((1140, 912), np.array(AFFINE, np.float32), ["a"])
    return aq, slm


def pattern(t):
    return CameraPattern("a", np.ones((60, 80), np.float32), t_index=t)


def ask(aq, slm, t):
    event = UpdatePatternEvent("a", index={"t": t, "p": "a"})
    slm.handle_message(UpdatePatternEventMessage(event))
    reply = aq.slm_to_microscope.get_nowait()
    assert reply.event_id == event.id
    return reply.pattern_t


def test_schedule_only_patterns_made_ahead_are_answered_for_their_own_timepoint():
    """The Manager is at t = 5 and the microscope is still at t = 2: the
    pattern of t = 2 is the answer, not the newest one."""
    aq, slm = make_buffer()
    for t in range(6):
        slm.handle_data(pattern(t))
    assert [ask(aq, slm, t) for t in (2, 3, 5)] == [2, 3, 5]


def test_image_based_patterns_are_answered_with_the_newest_one_finished():
    """Patterns made from the images of t = 0 and t = 5: every stimulation
    gets the newest one that exists when the microscope reaches it."""
    aq, slm = make_buffer()
    slm.handle_data(pattern(0))
    assert ask(aq, slm, 3) == 0
    slm.handle_data(pattern(5))
    assert ask(aq, slm, 4) == 0  # made for a later timepoint than the one run
    assert ask(aq, slm, 7) == 5


def test_only_later_patterns_kept_answers_with_the_oldest():
    aq, slm = make_buffer()
    slm.handle_data(pattern(5))
    slm.handle_data(pattern(6))
    assert ask(aq, slm, 2) == 5


def test_history_is_bounded_and_pruned():
    aq, slm = make_buffer()
    slm.max_history = 4
    for t in range(10):
        slm.handle_data(pattern(t))
    assert len(slm._history["a"]) == 4  # t = 6..9
    assert ask(aq, slm, 8) == 8
    assert [e[4] for e in slm._history["a"]] == [8, 9]


def test_event_without_a_timepoint_gets_the_newest_pattern():
    aq, slm = make_buffer()
    slm.handle_data(pattern(1))
    slm.handle_data(pattern(2))
    event = UpdatePatternEvent("a")
    slm.handle_message(UpdatePatternEventMessage(event))
    assert aq.slm_to_microscope.get_nowait().pattern_t == 2


def test_lagging_microscope_gets_the_pattern_current_when_it_arrives():
    """A pattern that arrives after the Manager scheduled the stimulation but
    before the microscope reached it is the one applied."""
    stop = Event()
    aq, slm = make_buffer(stop)
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))
    microscope = MicroscopeProcess(core, aq, stop_event=stop, settle_time_s=0.0)
    microscope.declare_slm()

    slm.handle_data(pattern(0))
    # the Manager schedules t = 6 now; the microscope is still busy elsewhere
    update = UpdatePatternEvent("a", index={"t": 6, "p": "a"})
    assert aq.microscope_to_slm_buffer.empty()
    # meanwhile the image of t = 5 is segmented and its pattern arrives
    slm.handle_data(pattern(5))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(slm.process)
        microscope.handle_update_pattern_event(update, slm_await_s=5.0)
        stop.set()
        future.result(timeout=5)

    assert microscope.current_pattern_t == 5
