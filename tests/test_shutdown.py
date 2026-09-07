"""
Shutdown-protocol tests for the five worker processes (everything except the
Manager): the graceful drain after the manager's close message, and the forced
stop via stop_event. Both must end with every thread exited and the outbox's
HDF5 files closed.
"""

from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event
from time import sleep

import numpy as np
from helpers import FakeImageSource, make_experiment, make_schedule

from pyclm.core.manager import MicroscopeOutbox, SLMBuffer
from pyclm.core.messages import CloseMessage
from pyclm.core.microscope import MicroscopeProcess
from pyclm.core.pattern_process import PatternProcess
from pyclm.core.queues import AllQueues
from pyclm.core.segmentation_process import SegmentationProcess
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def build_pipeline(tmp_path, stop):
    aq = AllQueues()
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))

    microscope = MicroscopeProcess(core, aq, stop_event=stop, settle_time_s=0.0)
    microscope.declare_slm()
    outbox = MicroscopeOutbox(aq, base_path=tmp_path, stop_event=stop)
    slm = SLMBuffer(aq, stop)
    seg = SegmentationProcess(aq, stop)
    pattern = PatternProcess(aq, stop)

    exp = make_experiment("exp.00")
    schedule = make_schedule([exp], steps=2)
    slm.initialize((8, 8), IDENTITY, ["exp.00"])
    outbox.initialize(schedule, core)
    assert outbox.open_files

    return aq, [microscope, outbox, slm, seg, pattern], outbox


def manager_close_queues(aq):
    return (
        aq.manager_to_microscope,
        aq.manager_to_outbox,
        aq.manager_to_slm_buffer,
        aq.manager_to_seg,
        aq.manager_to_pattern,
    )


def test_graceful_drain_after_manager_close(tmp_path):
    stop = Event()
    aq, processes, outbox = build_pipeline(tmp_path, stop)

    with ThreadPoolExecutor(max_workers=len(processes)) as executor:
        futures = [executor.submit(p.process) for p in processes]
        for queue in manager_close_queues(aq):
            queue.put(CloseMessage())
        _done, not_done = wait(futures, timeout=10)

    assert not not_done, "some processes did not exit after close"
    assert all(f.exception() is None for f in futures)
    assert outbox.open_files == {}
    assert not stop.is_set()
    assert all(p.error_count == 0 for p in processes)


def test_forced_stop_closes_files(tmp_path):
    stop = Event()
    _aq, processes, outbox = build_pipeline(tmp_path, stop)

    with ThreadPoolExecutor(max_workers=len(processes)) as executor:
        futures = [executor.submit(p.process) for p in processes]
        sleep(0.1)
        stop.set()
        _done, not_done = wait(futures, timeout=10)

    assert not not_done, "some processes did not exit after stop_event"
    assert all(f.exception() is None for f in futures)
    assert outbox.open_files == {}
