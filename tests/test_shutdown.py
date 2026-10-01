"""
Shutdown-protocol tests for the worker processes (everything except the
Manager): the graceful drain after the manager's close message, with the
stream close derived by the router rather than counted by each process, and
the forced stop via stop_event. Both must end with every thread exited and
the writer's outputs closed.
"""

from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event
from time import sleep

import numpy as np
from helpers import FakeImageSource, make_experiment, make_plan, make_schedule

from pyclm.core.experiments import TrackingConfig
from pyclm.core.manager import SLMBuffer
from pyclm.core.messages import CloseMessage
from pyclm.core.microscope import MicroscopeProcess
from pyclm.core.pattern_process import PatternProcess
from pyclm.core.patterns import ROI, CameraProperties, PatternMethod
from pyclm.core.queues import AllQueues
from pyclm.core.router import Router
from pyclm.core.segmentation import SegmentationMethod
from pyclm.core.segmentation_process import SegmentationProcess
from pyclm.core.tracking_process import TrackingProcess
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore
from pyclm.core.writer_process import WriterProcess

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


class StubSegmentation(SegmentationMethod):
    name = "stub_seg"

    def segment(self, data):
        return (data > 0).astype(np.uint16)


class StubPattern(PatternMethod):
    name = "stub"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_requirement("545", seg=True)

    def generate(self, context):
        return np.zeros(self.pattern_shape, dtype=np.float32)


class TrackStubPattern(StubPattern):
    name = "track_stub"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._requirements_list.clear()
        self.add_requirement("545", tracks=True)


def build_pipeline(tmp_path, stop, closed_loop=False, tracking=False):
    """Wire the five workers the way Controller.initialize does."""
    aq = AllQueues()
    core = SimulatedMicroscopeCore(FakeImageSource((16, 16)))

    microscope = MicroscopeProcess(core, aq, stop_event=stop, settle_time_s=0.0)
    microscope.declare_slm()
    writer = WriterProcess(base_path=tmp_path, stop_event=stop)
    slm = SLMBuffer(aq, stop)
    seg = SegmentationProcess(stop)
    track = TrackingProcess(stop)
    pattern = PatternProcess(aq, stop)

    if closed_loop or tracking:
        seg.register_method(StubSegmentation)
        pattern.register_method(StubPattern)
        pattern.register_method(TrackStubPattern)
        exp = make_experiment(
            "exp.00",
            segmentation_method="stub_seg",
            pattern_method="track_stub" if tracking else "stub",
        )
        if tracking:
            exp.tracking = TrackingConfig("centroid")
    else:
        exp = make_experiment("exp.00")
    schedule = make_schedule([exp], steps=2)

    pattern.initialize(CameraProperties(ROI(0, 0, 16, 16), 1.0))
    requirements = {"exp.00": pattern.request_method(exp)}
    pattern.initialize_models()
    plan = make_plan(schedule, requirements)

    router = Router(plan)
    for proc in (microscope, writer, seg, track, pattern):
        router.add(proc)
    router.resolve()
    for name in sorted({e for e, _ in router.demanded("seg")}):
        seg.request_method(schedule.experiments[name])
    for name, channel in sorted(router.demanded("tracks")):
        track.request_method(schedule.experiments[name], channel)

    slm.initialize((8, 8), IDENTITY, ["exp.00"])
    writer.initialize(plan, core)
    assert writer.writer.is_open

    processes = [slm, *router.active_processes()]
    return aq, processes, writer, router


def manager_close_queues(aq):
    return (aq.manager_to_microscope, aq.manager_to_slm_buffer, aq.manager_to_pattern)


def run_until_closed(aq, processes):
    with ThreadPoolExecutor(max_workers=len(processes)) as executor:
        futures = [executor.submit(p.process) for p in processes]
        for queue in manager_close_queues(aq):
            queue.put(CloseMessage())
        _done, not_done = wait(futures, timeout=10)
    return futures, not_done


def test_graceful_drain_after_manager_close(tmp_path):
    stop = Event()
    aq, processes, writer, router = build_pipeline(tmp_path, stop)

    # open loop: segmentation is never started
    assert [p.name for p in processes] == [
        "slm buffer",
        "microscope",
        "writer",
        "pattern",
    ]
    assert router.upstreams("pattern") == {"microscope"}

    futures, not_done = run_until_closed(aq, processes)

    assert not not_done, "some processes did not exit after close"
    assert all(f.exception() is None for f in futures)
    assert not writer.writer.is_open
    assert not stop.is_set()
    assert all(p.error_count == 0 for p in processes)


def test_graceful_drain_with_segmentation_in_the_loop(tmp_path):
    stop = Event()
    aq, processes, writer, router = build_pipeline(tmp_path, stop, closed_loop=True)

    assert [p.name for p in processes] == [
        "slm buffer",
        "microscope",
        "writer",
        "segmentation",
        "pattern",
    ]
    assert router.upstreams("segmentation") == {"microscope"}
    assert router.upstreams("pattern") == {"microscope", "segmentation"}
    assert router.upstreams("writer") == {"microscope", "segmentation"}

    futures, not_done = run_until_closed(aq, processes)

    assert not not_done, "some processes did not exit after close"
    assert all(f.exception() is None for f in futures)
    assert not writer.writer.is_open
    assert all(p.error_count == 0 for p in processes)
    assert all(getattr(p, "stream_ended", True) for p in processes)


def test_graceful_drain_with_tracking_in_the_loop(tmp_path):
    stop = Event()
    aq, processes, writer, router = build_pipeline(tmp_path, stop, tracking=True)

    assert [p.name for p in processes] == [
        "slm buffer",
        "microscope",
        "writer",
        "segmentation",
        "tracking",
        "pattern",
    ]
    assert router.upstreams("tracking") == {"microscope", "segmentation"}
    assert router.upstreams("pattern") == {"microscope", "tracking"}
    assert router.upstreams("writer") == {"microscope", "segmentation", "tracking"}

    futures, not_done = run_until_closed(aq, processes)

    assert not not_done, "some processes did not exit after close"
    assert all(f.exception() is None for f in futures)
    assert not writer.writer.is_open
    assert all(p.error_count == 0 for p in processes)


def test_forced_stop_closes_files(tmp_path):
    stop = Event()
    _aq, processes, writer, _router = build_pipeline(tmp_path, stop, closed_loop=True)

    with ThreadPoolExecutor(max_workers=len(processes)) as executor:
        futures = [executor.submit(p.process) for p in processes]
        sleep(0.1)
        stop.set()
        _done, not_done = wait(futures, timeout=10)

    assert not not_done, "some processes did not exit after stop_event"
    assert all(f.exception() is None for f in futures)
    assert not writer.writer.is_open
