"""
Named segmentations: [segmentation.<name>] tables, ``seg="<name>"``
requirements, ``seg:<name>`` routing at its own cadence, and the pattern
context reading several segmentations of one channel.
"""

from typing import ClassVar

import numpy as np
import pytest
from helpers import drain, make_experiment, make_plan, make_schedule

from pyclm.core.datatypes import AcquisitionData, SegmentationData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.experiments import SegmentationConfig, TrackingConfig
from pyclm.core.kinds import base_kind, seg_kind, seg_name
from pyclm.core.pattern_process import PatternProcess
from pyclm.core.patterns import (
    AcquiredImageRequest,
    DataDock,
    PatternContext,
    PatternMethod,
)
from pyclm.core.queues import AllQueues
from pyclm.core.router import Router
from pyclm.core.segmentation import SegmentationMethod
from pyclm.core.segmentation_process import SegmentationProcess
from pyclm.core.tracking_process import TrackingProcess
from pyclm.core.writer_process import WriterProcess
from pyclm.directories import experiment_from_toml

TOML = """
[imaging]
exposure = 10

[channels]
group = "Channel"
presets = ["545"]

[stimulation]
exposure = 10

[segmentation]
method = "cellpose"

[segmentation.nuclei]
method = "cellpose"
model = "nuclei"
save = false

[tracking]
method = "centroid"
segmentation = "nuclei"
max_distance_um = 12.5

[pattern]
method = "full_on"
"""


def event_for(exp, t, channel="545"):
    return AcquisitionEvent(
        exp.experiment_name,
        None,
        exp.channels[channel].channel_id,
        index={"t": t, "p": exp.experiment_name, "c": channel},
    )


# --------------------------------------------------------------- vocabulary
def test_kinds_vocabulary():
    assert seg_kind(None) == seg_kind("segmentation") == "seg"
    assert seg_kind("nuclei") == "seg:nuclei"
    assert seg_name("seg") == "segmentation"
    assert seg_name("seg:nuclei") == "nuclei"
    assert base_kind("seg:nuclei") == "seg"
    assert base_kind("tracks") == "tracks"
    with pytest.raises(ValueError, match="not a segmentation kind"):
        seg_name("raw")


# ------------------------------------------------------------------- config
def test_named_segmentation_tables_are_parsed(tmp_path):
    toml = tmp_path / "exp.toml"
    toml.write_text(TOML)
    exp = experiment_from_toml(toml, "exp")

    assert exp.segmentation.method_name == "cellpose"
    assert exp.segmentations["nuclei"].method_name == "cellpose"
    assert exp.segmentations["nuclei"].kwargs == {"model": "nuclei"}
    assert exp.segmentations["nuclei"].save is False
    assert exp.segmentation_names == ["segmentation", "nuclei"]
    assert exp.tracking.segmentation == "nuclei"
    assert exp.tracking.kwargs == {"max_distance_um": 12.5}
    assert exp.as_dict()["segmentations"]["nuclei"]["kwargs"] == {"model": "nuclei"}
    assert exp.as_dict()["tracking"]["segmentation"] == "nuclei"

    # named tables only: the default one is "none"
    toml.write_text(TOML.replace('[segmentation]\nmethod = "cellpose"\n', ""))
    exp = experiment_from_toml(toml, "exp")
    assert exp.segmentation.method_name == "none"
    assert exp.segmentation_names == ["nuclei"]

    toml.write_text(
        TOML.replace(
            '[segmentation.nuclei]\nmethod = "cellpose"\n', "[segmentation.nuclei]\n"
        )
    )
    with pytest.raises(
        ValueError,
        match=r"\[segmentation.nuclei\] is missing the required key 'method'",
    ):
        experiment_from_toml(toml, "exp")


def test_experiment_keeps_a_default_segmentation_and_a_setter():
    exp = make_experiment("exp.00")
    assert exp.segmentation.method_name == "none"
    assert exp.segmentation_names == []
    exp.segmentation = SegmentationConfig("cellpose")
    exp.segmentations["cyto"] = SegmentationConfig("cellpose", model="cyto3")
    assert exp.segmentation_names == ["segmentation", "cyto"]
    assert exp.segmentations["segmentation"] is exp.segmentation


# ------------------------------------------------------- requirements/dock
def test_requirements_name_segmentations_and_the_context_reads_them():
    class Both(PatternMethod):
        name = "both"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_requirement("545", raw=True, seg=["segmentation", "nuclei"])

        def generate(self, context):
            return np.zeros(self.pattern_shape, np.float32)

    exp = make_experiment("exp.00", segmentation_method="cellpose")
    exp.segmentations["nuclei"] = SegmentationConfig("cellpose", model="nuclei")
    cid = exp.channels["545"].channel_id

    reqs = Both().initialize(exp)
    assert reqs[0].needs_seg
    assert reqs[0].segmentations == ("nuclei",)
    assert reqs[0].kinds == ("raw", "seg", "seg:nuclei")

    only_named = PatternMethod()
    only_named.add_requirement("545", seg="nuclei")
    req = only_named.initialize(exp)[0]
    assert not req.needs_seg
    assert req.kinds == ("seg:nuclei",)
    assert AcquiredImageRequest(cid, False, True, True).kinds == ("seg", "tracks")

    dock = DataDock(1.0, reqs)
    assert set(dock.data[cid]) == {"raw", "seg", "seg:nuclei"}
    event = event_for(exp, 0)
    dock.add(AcquisitionData(event, np.full((8, 8), 7, np.uint16)))
    dock.add(SegmentationData(event, np.ones((8, 8), np.uint16)))
    assert not dock.check_complete()
    nuclei = SegmentationData(event, np.full((8, 8), 2, np.uint16), "nuclei")
    assert nuclei.kind == "seg:nuclei"
    assert nuclei.name == "nuclei"
    dock.add(nuclei)
    assert dock.check_complete()

    context = PatternContext(dock, exp, t=0)
    assert context.segmentation("545").max() == 1
    assert context.segmentation("545", "nuclei").max() == 2
    assert context.regions("545", "nuclei").ids.tolist() == [2]
    assert context.history("545", name="nuclei")[0].max() == 2
    assert context.history("545", kind="seg:nuclei")[0].max() == 2
    with pytest.raises(ValueError, match="not requested"):
        context.segmentation("545", "cyto")


def test_unconfigured_named_segmentation_is_a_warning_at_initialize(caplog):
    exp = make_experiment("exp.00")
    method = PatternMethod()
    method.add_requirement("545", seg="nuclei")
    import logging

    with caplog.at_level(logging.WARNING):
        method.initialize(exp)
    assert "does not configure" in caplog.text


# ------------------------------------------------------------------- router
class SegA(SegmentationMethod):
    name = "a"

    def segment(self, data):
        return (data > 0).astype(np.uint16)


class SegB(SegmentationMethod):
    name = "b"

    def segment(self, data):
        return (data > 0).astype(np.uint16) * 2


class Camera:
    name = "microscope"
    produces: ClassVar[dict] = {"raw": ()}
    always_active = True

    def attach(self, router, inbox):
        pass


def test_router_serves_two_segmentations_of_one_channel_at_their_own_cadence():
    exp = make_experiment(
        "exp.00", segmentation_method="a", pattern_kwargs={"every_t": 2}
    )
    exp.segmentations["nuclei"] = SegmentationConfig("b")
    exp.tracking = TrackingConfig("centroid", segmentation="nuclei")
    cid = exp.channels["545"].channel_id
    # the pattern wants the default segmentation at its cadence, and tracks:
    # tracking links the "nuclei" segmentation and needs every frame of it
    reqs = [AcquiredImageRequest(cid, True, True, True)]
    plan = make_plan(make_schedule([exp], steps=4), {"exp.00": reqs})

    seg = SegmentationProcess()
    seg.register_method(SegA, "a")
    seg.register_method(SegB, "b")
    router = Router(plan)
    for proc in (
        Camera(),
        WriterProcess(base_path=None),
        seg,
        TrackingProcess(),
        PatternProcess(AllQueues()),
    ):
        router.add(proc)
    router.resolve()

    routes = router.as_dict()["routes"]["exp.00"]["545"]
    assert routes["raw"] == ["pattern@pattern", "segmentation", "writer(record)"]
    assert routes["seg"] == ["pattern@pattern", "writer(record)"]
    assert routes["seg:nuclei"] == ["tracking", "writer(record)"]
    assert routes["tracks"] == ["pattern@pattern", "writer(record)"]
    assert router.demanded_kinds("seg") == {
        ("exp.00", "545", "seg"),
        ("exp.00", "545", "seg:nuclei"),
    }
    assert router.demanded("seg") == {("exp.00", "545")}
    assert router.produced_by("segmentation") == {
        ("exp.00", "545"): [("seg", "pattern"), ("seg:nuclei", "always")]
    }
    assert seg.segmentations_for("exp.00", "545") == [
        ("segmentation", "pattern"),
        ("nuclei", "always"),
    ]

    seg.request_method(exp)
    seg.request_method(exp, "nuclei")
    assert set(seg.models) == {("exp.00", "segmentation"), ("exp.00", "nuclei")}

    frame = np.ones((8, 8), np.uint16)
    seg.handle_data(AcquisitionData(event_for(exp, 0), frame))  # pattern due
    seg.handle_data(AcquisitionData(event_for(exp, 1), frame))  # not due

    to_pattern = drain(router.inbox("pattern"))
    assert [(d.kind, d.event.t_index) for d in to_pattern] == [("seg", 0)]
    to_tracking = drain(router.inbox("tracking"))
    assert [(d.name, d.event.t_index) for d in to_tracking] == [
        ("nuclei", 0),
        ("nuclei", 1),
    ]
    assert to_tracking[0].data.max() == 2
    to_writer = drain(router.inbox("writer"))
    assert [(d.kind, d.event.t_index) for d in to_writer] == [
        ("seg", 0),
        ("seg:nuclei", 0),
        ("seg:nuclei", 1),
    ]


def test_tracking_a_missing_named_segmentation_is_a_routing_error():
    from pyclm.core.router import RoutingError

    exp = make_experiment("exp.00", segmentation_method="a")
    exp.tracking = TrackingConfig("centroid", segmentation="nuclei")
    cid = exp.channels["545"].channel_id
    plan = make_plan(
        make_schedule([exp]),
        {"exp.00": [AcquiredImageRequest(cid, False, False, True)]},
    )
    router = Router(plan)
    for proc in (
        Camera(),
        WriterProcess(base_path=None),
        SegmentationProcess(),
        TrackingProcess(),
        PatternProcess(AllQueues()),
    ):
        router.add(proc)
    with pytest.raises(RoutingError, match="no method configured"):
        router.resolve()


def test_segmentation_process_without_a_router_runs_every_model():
    exp = make_experiment("exp.00", segmentation_method="a")
    exp.segmentations["nuclei"] = SegmentationConfig("b")
    seg = SegmentationProcess()
    seg.register_method(SegA, "a")
    seg.register_method(SegB, "b")
    seg.request_method(exp)
    seg.request_method(exp, "nuclei")

    class Sink:
        def __init__(self):
            self.items = []

        def publish(self, data):
            self.items.append(data)
            return 1

    sink = Sink()
    seg.attach(sink, None)
    seg.handle_data(AcquisitionData(event_for(exp, 0), np.ones((4, 4), np.uint16)))
    assert [(d.kind, int(d.data.max())) for d in sink.items] == [
        ("seg", 1),
        ("seg:nuclei", 2),
    ]
