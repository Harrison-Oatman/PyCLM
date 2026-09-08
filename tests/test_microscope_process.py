"""
Unit tests for MicroscopeProcess against the simulated core: frame delivery
to the router, configurable settle time, the per-message error guard, and
the SLM handshake.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter
from uuid import uuid4

import numpy as np
import pytest
from helpers import FakeImageSource, make_experiment

from pyclm.core.datatypes import AcquisitionData, EventSLMPattern
from pyclm.core.events import (
    AcquisitionEvent,
    UpdatePatternEvent,
    UpdateStagePositionEvent,
)
from pyclm.core.experiments import DeviceProperty, MicroscopePosition
from pyclm.core.messages import (
    AcquisitionEventMessage,
    CloseMessage,
    UpdatePositionEventMessage,
    UpdateZPositionMessage,
)
from pyclm.core.microscope import MicroscopeProcess
from pyclm.core.position_mover import PositionMover
from pyclm.core.queues import AllQueues
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore

EXP = make_experiment("exp.00")


class Sink:
    """Stands in for the Router: records what the microscope publishes."""

    def __init__(self):
        self.items = []
        self.ended = []

    def publish(self, data):
        self.items.append(data)
        return 1

    def end_stream(self, name):
        self.ended.append(name)


def make_microscope(**kwargs):
    aq = AllQueues()
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))
    microscope = MicroscopeProcess(
        core, aq, stop_event=Event(), settle_time_s=0.0, **kwargs
    )
    microscope.declare_slm()
    microscope.attach(Sink())
    return aq, microscope, core


def make_event(t_index=0, devices=None):
    return AcquisitionEvent(
        EXP.experiment_name,
        MicroscopePosition(0.0, 0.0, 0.0, label="exp.00"),
        EXP.channels["545"].channel_id,
        index={"t": t_index, "p": "exp.00", "c": "545"},
        exposure_time_ms=5,
        devices=devices,
    )


def test_acquisition_event_delivers_frame():
    _aq, microscope, _ = make_microscope()
    event = make_event()

    microscope.handle_acquisition_event(event)

    [data] = microscope.router.items
    assert isinstance(data, AcquisitionData)
    assert data.data.shape == (16, 16)
    assert data.event is event
    assert event.completed_time is not None


@pytest.mark.parametrize(("settle", "minimum"), [(0.0, 0.0), (0.2, 0.2)])
def test_settle_time_is_configurable(settle, minimum):
    _aq, microscope, _ = make_microscope()
    microscope.settle_time_s = settle

    start = perf_counter()
    microscope.handle_acquisition_event(make_event())
    elapsed = perf_counter() - start

    assert elapsed >= minimum
    assert elapsed < minimum + 0.15


def test_handler_error_is_counted_and_process_continues():
    aq, microscope, _ = make_microscope()
    bad = make_event(devices=[DeviceProperty("Cam", "Gain", "1", "no_such_type")])
    good = make_event(t_index=1)

    aq.manager_to_microscope.put(AcquisitionEventMessage(bad))
    aq.manager_to_microscope.put(AcquisitionEventMessage(good))
    aq.manager_to_microscope.put(CloseMessage())

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(microscope.process)
        assert future.result(timeout=5) == 0

    assert microscope.error_count == 1
    assert microscope.consecutive_errors == 0

    delivered = microscope.router.items
    assert [type(d) for d in delivered] == [AcquisitionData]
    assert delivered[0].event is good
    assert microscope.router.ended == ["microscope"]


def test_consecutive_errors_abort_the_run():
    aq, microscope, _ = make_microscope(max_consecutive_errors=2)
    for _ in range(2):
        aq.manager_to_microscope.put(
            AcquisitionEventMessage(
                make_event(devices=[DeviceProperty("Cam", "Gain", "1", "bad")])
            )
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(microscope.process)
        exc = future.exception(timeout=5)

    assert isinstance(exc, KeyError)
    assert microscope.error_count == 2


def test_slm_handshake_skips_stale_replies():
    aq, microscope, core = make_microscope()
    update = UpdatePatternEvent("exp.00")
    stale = EventSLMPattern(uuid4(), np.zeros((8, 8), np.uint8), "stale")
    good = EventSLMPattern(update.id, np.ones((8, 8), np.uint8), "good")
    aq.slm_to_microscope.put(stale)
    aq.slm_to_microscope.put(good)

    microscope.handle_update_pattern_event(update, slm_await_s=1.0)

    assert microscope.current_pattern_id == "good"
    assert core._slm_image is not None
    assert core._slm_image.min() == 1
    assert aq.slm_to_microscope.empty()


def test_slm_handshake_timeout_keeps_current_pattern(caplog):
    _aq, microscope, _ = make_microscope()
    microscope.current_pattern_id = "previous"
    update = UpdatePatternEvent("exp.00")

    with caplog.at_level(logging.WARNING):
        start = perf_counter()
        microscope.handle_update_pattern_event(update, slm_await_s=0.05)
        elapsed = perf_counter() - start

    assert microscope.current_pattern_id == "previous"
    assert 0.04 <= elapsed < 1.0
    assert "keeping the current pattern" in caplog.text


def test_position_update_reports_z_correction():
    class OffsetMover(PositionMover):
        def move_to(self, position, core):
            core.setXYPosition(position.x, position.y)
            core.setPosition(position.z + 3.0)
            return True, core.getZPosition()

    aq = AllQueues()
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))
    microscope = MicroscopeProcess(
        core, aq, position_mover=OffsetMover(), stop_event=Event()
    )
    position = MicroscopePosition(1.0, 2.0, 10.0, label="exp.00")

    microscope.handle_message(
        UpdatePositionEventMessage(UpdateStagePositionEvent(position, "exp.00"))
    )

    msg = aq.microscope_to_manager.get_nowait()
    assert isinstance(msg, UpdateZPositionMessage)
    assert msg.new_z_position == 13.0
    assert msg.experiment_name == "exp.00"
