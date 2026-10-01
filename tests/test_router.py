"""
Router tests: resolution of the subscription table from what processes
demand and produce, cadence filtering on publish, and the derived stream
close (docs/stage3-router-design.md §3).
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import pytest
from helpers import drain, make_experiment, make_plan, make_schedule

from pyclm.core.datatypes import AcquisitionData, SegmentationData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.messages import StreamCloseMessage
from pyclm.core.patterns import AcquiredImageRequest
from pyclm.core.router import Router, RoutingError, Subscription


class Stub:
    """The minimum the router needs from a process."""

    def __init__(
        self,
        name,
        produces=None,
        subs=(),
        always_active=False,
        can=True,
        continuous=False,
    ):
        self.name = name
        self.produces = dict(produces or {})
        self._subs = list(subs)
        self.always_active = always_active
        self.continuous = continuous
        self._can = can
        self.router = None
        self.inbox: Queue | None = None

    def subscriptions(self, plan):
        return list(self._subs)

    def can_produce(self, kind, experiment, channel):
        return self._can and kind in self.produces

    def attach(self, router, inbox):
        self.router = router
        self.inbox = inbox

    def received(self):
        return drain(self.inbox) if self.inbox is not None else []


def sub(consumer, channel, kind, cadence="always", demand=True, exp="exp.00"):
    return Subscription(consumer, exp, channel, kind, cadence, demand)


def writer_subs(plan, seg=True, tracks=False, exp="exp.00"):
    out = []
    for ch in plan.channels(exp):
        out.append(sub("writer", ch, "raw", demand=False, exp=exp))
        if seg:
            out.append(sub("writer", ch, "seg", demand=False, exp=exp))
        if tracks:
            out.append(sub("writer", ch, "tracks", demand=False, exp=exp))
    return out


def build(plan, *extra, seg_can=True, writer_seg=True, writer_tracks=False):
    microscope = Stub("microscope", {"raw": ()}, always_active=True)
    writer = Stub(
        "writer", subs=writer_subs(plan, writer_seg, writer_tracks), always_active=True
    )
    segmentation = Stub("segmentation", {"seg": ("raw",)}, can=seg_can)
    router = Router(plan)
    procs = {"microscope": microscope, "writer": writer, "segmentation": segmentation}
    for proc in (microscope, writer, segmentation, *extra):
        router.add(proc)
        procs[proc.name] = proc
    return router, procs


def routes(router):
    return router.as_dict()["routes"]["exp.00"]


def frame(plan, t, channel, kind="raw", exp="exp.00"):
    cfg = plan.imaging_config(exp, channel)
    event = AcquisitionEvent(
        exp,
        plan.schedule.positions[exp],
        cfg.channel_id,
        index={"t": t, "p": exp, "c": channel},
    )
    img = np.zeros((4, 4), np.uint16)
    return (
        SegmentationData(event, img) if kind == "seg" else AcquisitionData(event, img)
    )


# ------------------------------------------------------------------ resolve
def test_pattern_needing_segmentation_pulls_in_the_producer():
    exp = make_experiment("exp.00", segmentation_method="cellpose")
    plan = make_plan(make_schedule([exp], steps=2))
    pattern = Stub(
        "pattern", subs=[sub("pattern", "545", "seg", "pattern")], always_active=True
    )
    router, procs = build(plan, pattern)
    router.resolve()

    assert routes(router) == {
        "545": {
            "raw": ["segmentation@pattern", "writer(record)"],
            "seg": ["pattern@pattern", "writer(record)"],
        },
        "DMD": {"raw": ["writer(record)"]},
    }
    assert router.demanded("seg") == {("exp.00", "545")}
    assert router.demanded("raw") == {("exp.00", "545")}
    assert [p.name for p in router.active_processes()] == [
        "microscope",
        "writer",
        "segmentation",
        "pattern",
    ]
    assert router.upstreams("segmentation") == {"microscope"}
    assert router.upstreams("pattern") == {"microscope", "segmentation"}
    assert router.upstreams("writer") == {"microscope", "segmentation"}
    assert router.upstreams("microscope") == set()
    assert router.deliveries_to("writer") == {
        "raw": {("exp.00", "545"), ("exp.00", "DMD")},
        "seg": {("exp.00", "545")},
    }
    assert all(procs[n].inbox is not None for n in router.as_dict()["active"])


def test_open_loop_pattern_leaves_segmentation_inactive():
    exp = make_experiment("exp.00", segmentation_method="cellpose")
    plan = make_plan(make_schedule([exp], steps=2))
    pattern = Stub("pattern", always_active=True)
    router, _ = build(plan, pattern)
    router.resolve()

    # the writer's seg subscription records only what is produced: nothing here
    assert "seg" not in routes(router)["545"]
    assert not router.is_active("segmentation")
    assert router.demanded("seg") == set()
    assert router.upstreams("pattern") == {"microscope"}
    assert router.upstreams("writer") == {"microscope"}


def test_tracking_widens_segmentation_to_every_frame():
    exp = make_experiment("exp.00", segmentation_method="cellpose")
    plan = make_plan(make_schedule([exp], steps=2))
    pattern = Stub(
        "pattern", subs=[sub("pattern", "545", "tracks", "pattern")], always_active=True
    )
    # tracking is stateful, so it declares it needs every frame of its inputs
    tracking = Stub("tracking", {"tracks": ("seg",)}, continuous=True)
    router, _ = build(plan, pattern, tracking, writer_tracks=True)
    router.resolve()

    assert routes(router)["545"] == {
        "raw": ["segmentation", "writer(record)"],
        "seg": ["tracking", "writer(record)"],
        "tracks": ["pattern@pattern", "writer(record)"],
    }
    assert router.upstreams("tracking") == {"microscope", "segmentation"}
    assert router.upstreams("pattern") == {"microscope", "tracking"}
    assert router.upstreams("writer") == {"microscope", "segmentation", "tracking"}


def test_two_consumers_widen_a_shared_producer():
    exp = make_experiment("exp.00", segmentation_method="cellpose")
    plan = make_plan(make_schedule([exp], steps=2))
    pattern = Stub(
        "pattern", subs=[sub("pattern", "545", "seg", "pattern")], always_active=True
    )
    monitor = Stub("monitor", subs=[sub("monitor", "545", "seg", "always")])
    router, _ = build(plan, pattern, monitor)
    router.resolve()

    assert routes(router)["545"]["raw"] == ["segmentation", "writer(record)"]
    assert routes(router)["545"]["seg"] == [
        "pattern@pattern",
        "monitor",
        "writer(record)",
    ]


def test_missing_producer_is_an_error():
    exp = make_experiment("exp.00")
    plan = make_plan(make_schedule([exp], steps=2))
    pattern = Stub("pattern", subs=[sub("pattern", "545", "tracks", "pattern")])
    router, _ = build(plan, pattern)

    with pytest.raises(RoutingError, match="no registered process produces 'tracks'"):
        router.resolve()


def test_producer_without_a_configured_method_is_an_error():
    exp = make_experiment("exp.00")  # segmentation "none"
    plan = make_plan(make_schedule([exp], steps=2))
    pattern = Stub("pattern", subs=[sub("pattern", "545", "seg", "pattern")])
    router, _ = build(plan, pattern, seg_can=False)

    with pytest.raises(RoutingError, match="no method configured"):
        router.resolve()


def test_unknown_channel_and_kind_are_errors():
    exp = make_experiment("exp.00")
    plan = make_plan(make_schedule([exp], steps=2))

    router, _ = build(plan, Stub("p", subs=[sub("p", "nope", "raw")]))
    with pytest.raises(RoutingError, match="no channel 'nope'"):
        router.resolve()

    router, _ = build(plan, Stub("p", subs=[sub("p", "545", "labels")]))
    with pytest.raises(RoutingError, match="unknown data kind"):
        router.resolve()


def test_pattern_cadence_on_a_channel_absent_when_due_is_an_error():
    # pattern every 2 needing only the stimulation frame (lcm 2); channel 545
    # every 3 is not acquired at t=2 although the pattern is due then
    exp = make_experiment("exp.00", every_t=3, pattern_kwargs={"every_t": 2})
    reqs = {"exp.00": [AcquiredImageRequest(exp.stimulation.channel_id, True, False)]}
    plan = make_plan(make_schedule([exp], steps=4), reqs)
    user = Stub("user", subs=[sub("user", "545", "raw", "pattern")])
    router, _ = build(plan, user)

    with pytest.raises(RoutingError, match="t=2"):
        router.resolve()


def test_registration_errors():
    exp = make_experiment("exp.00")
    plan = make_plan(make_schedule([exp], steps=2))
    router, _ = build(plan)

    with pytest.raises(RoutingError, match="already registered"):
        router.add(Stub("writer"))
    with pytest.raises(RoutingError, match="both produce 'raw'"):
        router.add(Stub("camera2", {"raw": ()}))
    router.resolve()
    with pytest.raises(RoutingError, match="after the router is resolved"):
        router.add(Stub("late"))
    with pytest.raises(RoutingError, match="already resolved"):
        router.resolve()


# ------------------------------------------------------------------ publish
def test_publish_delivers_the_same_object_at_the_right_cadence():
    exp = make_experiment(
        "exp.00", segmentation_method="cellpose", pattern_kwargs={"every_t": 2}
    )
    plan = make_plan(make_schedule([exp], steps=4))
    pattern = Stub(
        "pattern", subs=[sub("pattern", "545", "seg", "pattern")], always_active=True
    )
    router, procs = build(plan, pattern)
    router.resolve()

    raw0 = frame(plan, 0, "545")
    assert router.publish(raw0) == 2  # segmentation (pattern due) + writer
    raw1 = frame(plan, 1, "545")
    assert router.publish(raw1) == 1  # writer only
    seg0 = frame(plan, 0, "545", kind="seg")
    assert router.publish(seg0) == 2

    [seg_in] = procs["segmentation"].received()
    assert seg_in is raw0
    assert procs["writer"].received() == [raw0, raw1, seg0]
    assert procs["pattern"].received() == [seg0]
    assert router.undeliverable == 0


def test_publish_without_subscriber_is_counted_and_dropped(caplog):
    exp = make_experiment("exp.00")
    plan = make_plan(make_schedule([exp], steps=2))
    router, procs = build(plan, Stub("pattern", always_active=True), writer_seg=False)
    router.resolve()

    with caplog.at_level(logging.WARNING):
        assert router.publish(frame(plan, 0, "545", kind="seg")) == 0
        assert router.publish(frame(plan, 1, "545", kind="seg")) == 0

    assert router.undeliverable == 2
    assert caplog.text.count("no subscriber") == 1
    assert procs["writer"].received() == []


# --------------------------------------------------------------- stream end
def test_stream_close_arrives_after_the_last_upstream():
    exp = make_experiment("exp.00", segmentation_method="cellpose")
    plan = make_plan(make_schedule([exp], steps=2))
    pattern = Stub(
        "pattern", subs=[sub("pattern", "545", "seg", "pattern")], always_active=True
    )
    router, procs = build(plan, pattern)
    router.resolve()

    assert router.end_stream("microscope") == ["segmentation"]
    assert isinstance(procs["segmentation"].received()[0], StreamCloseMessage)
    assert procs["pattern"].received() == []
    assert procs["writer"].received() == []

    assert router.end_stream("microscope") == []  # idempotent
    assert sorted(router.end_stream("segmentation")) == ["pattern", "writer"]
    for name in ("pattern", "writer"):
        [msg] = procs[name].received()
        assert isinstance(msg, StreamCloseMessage)
    assert procs["microscope"].received() == []


def test_stream_close_is_delivered_once_under_concurrent_producers():
    exp = make_experiment("exp.00", segmentation_method="cellpose")
    plan = make_plan(make_schedule([exp], steps=2))
    pattern = Stub(
        "pattern", subs=[sub("pattern", "545", "seg", "pattern")], always_active=True
    )
    tracking = Stub("tracking", {"tracks": ("seg",)})
    user = Stub("user", subs=[sub("user", "545", "tracks", "always")])
    router, procs = build(plan, pattern, tracking, user)
    router.resolve()

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(
            pool.map(
                router.end_stream,
                ["microscope", "segmentation", "tracking"] * 5,
            )
        )

    for name in ("segmentation", "tracking", "pattern", "writer", "user"):
        items = procs[name].received()
        assert [type(i) for i in items] == [StreamCloseMessage], name
