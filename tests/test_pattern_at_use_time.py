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


# ---------------------------------------------------------------------------
# render_at_stimulation: the method is run again when the microscope reaches a
# stimulation, on the newest inputs and with that stimulation's timepoint
# ---------------------------------------------------------------------------


def make_latest_process():
    from helpers import make_experiment
    from test_pattern_process import StubPattern

    from pyclm.core.pattern_process import PatternProcess
    from pyclm.core.patterns import ROI, CameraProperties

    class LatestStub(StubPattern):
        name = "latest_stub"
        render_at_stimulation = True

    exp = make_experiment("exp.00", pattern_method="latest_stub")
    aq = AllQueues()
    pp = PatternProcess(aq)
    pp.register_method(LatestStub, "latest_stub")
    pp.initialize(CameraProperties(ROI(0, 0, 64, 48), 1.0))
    requirements = pp.request_method(exp)
    pp.initialize_models()
    return exp, aq, pp, requirements


def test_stimulation_request_renders_from_the_newest_inputs():
    from test_pattern_process import make_raw

    from pyclm.core.pattern_process import RequestPattern

    exp, aq, pp, requirements = make_latest_process()
    assert pp.renders_at_stimulation() == {"exp.00"}

    pp.handle_message(RequestPattern(5, 300.0, "exp.00", requirements))
    pp.handle_data(make_raw(exp, t_index=5))
    assert aq.pattern_to_slm.get_nowait().t_index == 5

    update = UpdatePatternEvent(
        "exp.00", index={"t": 8, "p": "exp.00"}, scheduled_time_since_start=480.0
    )
    request = UpdatePatternEventMessage(update)
    pp.handle_stimulation_request(request)

    # the new pattern first, then the request: the buffer answers with it
    made = aq.pattern_to_slm.get_nowait()
    assert isinstance(made, CameraPattern)
    assert made.t_index == 5  # labeled with the timepoint of its inputs
    assert aq.pattern_to_slm.get_nowait() is request

    context = pp.models["exp.00"].contexts[-1]
    assert (context.t, context.time) == (8, 480.0)


def test_stimulation_request_before_any_input_is_only_passed_on():
    _exp, aq, pp, _ = make_latest_process()
    request = UpdatePatternEventMessage(
        UpdatePatternEvent("exp.00", index={"t": 0, "p": "exp.00"})
    )
    pp.handle_stimulation_request(request)
    assert aq.pattern_to_slm.get_nowait() is request
    assert aq.pattern_to_slm.empty()
    assert pp.models["exp.00"].contexts == []


def test_microscope_sends_the_request_through_the_pattern_process():
    aq = AllQueues()
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))
    microscope = MicroscopeProcess(core, aq, stop_event=Event(), settle_time_s=0.0)
    microscope.declare_slm()
    microscope.render_at_stimulation = {"a"}

    microscope.handle_update_pattern_event(UpdatePatternEvent("a"), slm_await_s=0.01)
    microscope.handle_update_pattern_event(UpdatePatternEvent("b"), slm_await_s=0.01)

    assert aq.microscope_to_pattern.get_nowait().event.experiment_name == "a"
    assert aq.microscope_to_slm_buffer.get_nowait().event.experiment_name == "b"
    assert aq.microscope_to_pattern.empty()
    assert aq.microscope_to_slm_buffer.empty()


def test_dry_run_lights_cells_between_images(tmp_path):
    """The demo's closed-loop experiment imaged every 2nd timepoint and
    stimulated at every one, with a method that renders at each stimulation."""
    import pyclm.io as pio
    from pyclm import cli, run_pyclm

    target = tmp_path / "demo"
    assert cli.main(["new", str(target), "--demo"]) == 0
    with open(target / "demo_methods.py", "a", encoding="utf-8") as f:
        f.write(
            "\n\nfrom pyclm.core.patterns.fbc_cell_movement import MoveOutModel\n\n\n"
            "class MoveOutLatest(MoveOutModel):\n"
            '    name = "move_out_latest"\n'
            "    render_at_stimulation = True\n"
        )
    toml = (target / "cells.toml").read_text(encoding="utf-8")
    toml = toml.replace('method = "move_out"', 'method = "move_out_latest"')
    toml = toml.replace(
        "every_t = 1                   # image every timepoint", "every_t = 2"
    )
    assert "every_t = 2" in toml
    (target / "cells.toml").write_text(toml, encoding="utf-8")
    (target / "schedule.toml").write_text(
        "[timing]\nsteps = 6\ninterval_seconds = 0.5\nsetup_time_seconds = 0.0\n"
        "time_between_positions = 0.05\n"
    )
    run_pyclm(target, dry=True)

    with pio.open(target / "cells.00.zarr") as exp:
        frames = exp.frames.to_pandas()
        lit = exp.camera_pattern_at(exp.current_t) > 0
    stim = frames[frames.kind == "stim_event"].sort_values("t")
    assert list(stim.t) == list(range(6))  # stimulated between images too
    assert lit.any()
    later = stim[stim.t >= 2]
    made_from = later.pattern_t.ffill()
    assert made_from.notna().all()
    assert (made_from <= later.t).all()
    assert (made_from % 2 == 0).all()  # made from images, taken at even t
