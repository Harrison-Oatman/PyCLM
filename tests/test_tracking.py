"""
Tracking: the built-in centroid linker on synthetic moving blobs, the
TrackingProcess as a router consumer/producer, the [tracking] TOML table,
and what a pattern method sees through context.tracks().
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
from helpers import make_experiment, make_plan, make_schedule

from pyclm.core.datatypes import SegmentationData, TrackingData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.experiments import TrackingConfig
from pyclm.core.patterns import AcquiredImageRequest, DataDock, PatternContext
from pyclm.core.router import Router
from pyclm.core.tracking import CentroidTracker, TrackingMethod, TrackRow, Tracks
from pyclm.core.tracking_process import TrackingProcess
from pyclm.directories import experiment_from_toml


def blobs(centres, shape=(64, 64), radius=4):
    """Label image with one disc per centre, labelled 1..n in the given order."""
    img = np.zeros(shape, dtype=np.uint16)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for label, (y, x) in enumerate(centres, start=1):
        img[(yy - y) ** 2 + (xx - x) ** 2 <= radius**2] = label
    return img


def ids_at(rows, centre, tol=1.0):
    return [
        r.track_id for r in rows if abs(r.y - centre[0]) + abs(r.x - centre[1]) < tol
    ]


# ------------------------------------------------------------- centroid tracker
def test_centroid_tracker_keeps_ids_under_small_motion():
    tracker = CentroidTracker("exp.00", "545", max_distance_um=5.0)

    labels0, rows0 = tracker.track(blobs([(10, 10), (40, 40)]), 0, 1.0)
    assert sorted(r.track_id for r in rows0) == [1, 2]
    assert labels0.dtype == np.uint32
    assert set(np.unique(labels0)) == {0, 1, 2}

    # both blobs move 2 px; the segmentation labels them in the opposite order
    labels1, rows1 = tracker.track(blobs([(42, 42), (12, 12)]), 1, 1.0)
    assert ids_at(rows1, (12, 12)) == [1]
    assert ids_at(rows1, (42, 42)) == [2]
    assert labels1[12, 12] == 1
    assert labels1[42, 42] == 2
    assert {r.label for r in rows1} == {
        1,
        2,
    }  # segmentation labels are kept in the rows


def test_centroid_tracker_starts_new_ids_beyond_max_distance_and_on_appearance():
    tracker = CentroidTracker("exp.00", "545", max_distance_um=5.0)
    tracker.track(blobs([(10, 10), (40, 40)]), 0, 1.0)

    # first blob jumps 20 px (a new object), second stays, a third appears
    _, rows = tracker.track(blobs([(30, 10), (40, 40), (55, 20)]), 1, 1.0)
    assert ids_at(rows, (40, 40)) == [2]
    assert ids_at(rows, (30, 10)) == [3]
    assert ids_at(rows, (55, 20)) == [4]

    # a disappearing object simply ends; the survivor keeps its id
    _, rows = tracker.track(blobs([(41, 41)]), 2, 1.0)
    assert [r.track_id for r in rows] == [2]


def test_centroid_tracker_distance_gate_is_in_micrometres():
    tracker = CentroidTracker("exp.00", "545", max_distance_um=5.0)
    tracker.track(blobs([(10, 10)]), 0, pixel_size_um=2.0)
    # 4 px = 8 um > 5 um: a new id
    _, rows = tracker.track(blobs([(14, 10)]), 1, pixel_size_um=2.0)
    assert rows[0].track_id == 2


def test_centroid_tracker_handles_empty_frames():
    tracker = CentroidTracker("exp.00", "545")
    labels, rows = tracker.track(np.zeros((8, 8), np.uint16), 0, 1.0)
    assert rows == []
    assert labels.shape == (8, 8)
    _, rows = tracker.track(blobs([(4, 4)], shape=(8, 8), radius=1), 1, 1.0)
    assert rows[0].track_id == 1


# --------------------------------------------------------------- Tracks object
def test_tracks_object_accessors():
    labels = blobs([(10, 10), (40, 40)])
    tracked = np.where(labels == 1, 7, np.where(labels == 2, 9, 0)).astype(np.uint32)
    rows = [TrackRow(7, 1, 10.0, 10.0, 49, 0), TrackRow(9, 2, 40.0, 40.0, 49, 7)]
    tracks = Tracks(tracked, rows)

    assert list(tracks.ids) == [7, 9]
    assert tracks.centroid(9) == (40.0, 40.0)
    assert tracks.centroid(8) is None
    assert tracks.mask(7).sum() == (labels == 1).sum()
    assert tracks.table.column_names == [
        "track_id",
        "label",
        "y",
        "x",
        "area",
        "parent",
    ]
    assert tracks.table.column("parent").to_pylist() == [0, 7]
    assert len(tracks) == 2


# ----------------------------------------------------------- tracking process
def seg_data(exp, t, labels):
    event = AcquisitionEvent(
        exp.experiment_name,
        None,
        exp.channels["545"].channel_id,
        index={"t": t, "p": exp.experiment_name, "c": "545"},
    )
    event.pixel_width_um = 1.0
    return SegmentationData(event, labels)


class Sink:
    def __init__(self):
        self.items = []

    def publish(self, data):
        self.items.append(data)
        return 1

    def end_stream(self, name):
        pass


def test_tracking_process_relabels_and_publishes():
    exp = make_experiment("exp.00", segmentation_method="cellpose")
    exp.tracking = TrackingConfig("centroid", max_distance_um=5.0)
    proc = TrackingProcess()
    assert proc.can_produce("tracks", exp, "545")
    proc.request_method(exp, "545")
    sink = Sink()
    proc.attach(sink, None)

    proc.handle_data(seg_data(exp, 0, blobs([(10, 10), (40, 40)])))
    proc.handle_data(seg_data(exp, 1, blobs([(42, 42), (12, 12)])))

    assert [type(d) for d in sink.items] == [TrackingData, TrackingData]
    first, second = sink.items
    assert first.kind == "tracks"
    assert first.event.t_index == 0
    assert second.labels[12, 12] == 1
    assert second.labels[42, 42] == 2
    assert [r.track_id for r in second.rows] == [2, 1]


def test_tracking_process_needs_a_configured_method():
    exp = make_experiment("exp.00")  # no [tracking]
    proc = TrackingProcess()
    assert not proc.can_produce("tracks", exp, "545")
    assert proc.produces == {"tracks": ("seg",)}
    assert proc.continuous


def test_custom_tracking_method_is_registered_per_instance():
    class Constant(TrackingMethod):
        name = "constant"

        def track(self, labels, t, pixel_size_um):
            return labels.astype(np.uint32), [TrackRow(1, 1, 0.0, 0.0, 1, 0)]

    proc = TrackingProcess()
    proc.register_method(Constant)
    assert "constant" in proc.known_methods
    assert "constant" not in TrackingProcess().known_methods


# ------------------------------------------------------------ TOML and router
def test_tracking_section_is_parsed(tmp_path: Path):
    toml = tmp_path / "exp.toml"
    toml.write_text(
        """
[imaging]
exposure = 10
[channels]
group = "Channel"
presets = ["545"]
[stimulation]
exposure = 5
[segmentation]
method = "cellpose"
[tracking]
method = "centroid"
max_distance_um = 12.5
save = false
[pattern]
method = "full_on"
"""
    )
    exp = experiment_from_toml(toml, "exp")
    assert exp.tracking.method_name == "centroid"
    assert exp.tracking.kwargs == {"max_distance_um": 12.5}
    assert exp.tracking.save is False
    assert exp.as_dict()["tracking"]["method_name"] == "centroid"

    toml.write_text(toml.read_text().replace('[tracking]\nmethod = "centroid"\n', ""))
    assert experiment_from_toml(toml, "exp").tracking.method_name == "none"


def test_router_runs_tracking_on_every_frame_when_a_pattern_wants_tracks():
    exp = make_experiment("exp.00", segmentation_method="cellpose")
    exp.tracking = TrackingConfig("centroid")
    reqs = [AcquiredImageRequest(exp.channels["545"].channel_id, False, False, True)]
    plan = make_plan(make_schedule([exp], steps=3), {"exp.00": reqs})

    from pyclm.core.pattern_process import PatternProcess
    from pyclm.core.queues import AllQueues
    from pyclm.core.segmentation_process import SegmentationProcess
    from pyclm.core.writer_process import WriterProcess

    class Camera:
        name = "microscope"
        produces: ClassVar[dict] = {"raw": ()}
        always_active = True

        def attach(self, router, inbox):
            pass

    router = Router(plan)
    for proc in (
        Camera(),
        WriterProcess(base_path=None),
        SegmentationProcess(),
        TrackingProcess(),
        PatternProcess(AllQueues()),
    ):
        router.add(proc)
    router.resolve()

    routes = router.as_dict()["routes"]["exp.00"]["545"]
    assert routes["raw"] == ["segmentation", "writer(record)"]
    assert routes["seg"] == ["tracking", "writer(record)"]
    assert routes["tracks"] == ["pattern@pattern", "writer(record)"]
    assert router.demanded("tracks") == {("exp.00", "545")}


# ---------------------------------------------------------- pattern context
def test_context_tracks_returns_the_tracked_labels_and_rows():
    exp = make_experiment("exp.00")
    cid = exp.channels["545"].channel_id
    dock = DataDock(3.0, [AcquiredImageRequest(cid, False, True, True)])
    labels = blobs([(10, 10)])
    event = AcquisitionEvent(
        "exp.00", None, cid, index={"t": 4, "p": "exp.00", "c": "545"}
    )
    dock.add(SegmentationData(event, labels))
    dock.add(
        TrackingData(
            event, labels.astype(np.uint32) * 5, [TrackRow(5, 1, 10.0, 10.0, 49)]
        )
    )
    assert dock.check_complete()

    context = PatternContext(dock, exp, t=4)
    tracks = context.tracks("545")
    assert isinstance(tracks, Tracks)
    assert tracks.labels[10, 10] == 5
    assert tracks.centroid(5) == (10.0, 10.0)
    assert context.segmentation("545")[10, 10] == 1
    assert context.t == 4
    assert context.time == 3.0
