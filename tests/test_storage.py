"""
Storage formats: cadence grouping, the OME-Zarr writer end to end, the
pyclm.io readers for both formats, and ImageJ export.
"""

import json
from threading import Event

import numpy as np
import pytest
import zarr
from helpers import FakeImageSource, make_experiment, make_plan, make_schedule

import pyclm.io as pio
from pyclm.core.datatypes import AcquisitionData, SegmentationData, StimulationData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.storage import HDF5WriterV1, OMEZarrWriter, cadence_groups, make_writer
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore
from pyclm.core.writer_process import WriterProcess

AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
SLM = (12, 10)


def core_with(shape=(16, 16)):
    core = SimulatedMicroscopeCore(FakeImageSource(shape), slm_shape=SLM)
    core._roi = (0, 0, shape[1], shape[0])
    core.getROI = lambda: (0, 0, shape[1], shape[0])
    return core


def acquire(plan, name, t, channel, image, pattern=None, pattern_id=None, camera=None):
    """Build the data object the microscope would deliver for one planned frame."""
    ev = next(
        e
        for e in plan.events_at(t)
        if e.kind == "acquire" and e.experiment == name and e.channel == channel
    )
    cfg = plan.imaging_config(name, channel)
    event = AcquisitionEvent(
        name,
        plan.schedule.positions[name],
        cfg.channel_id,
        index=ev.index,
        scheduled_time=1_700_000_000 + t,
        exposure_time_ms=cfg.exposure,
        needs_slm=ev.is_stim,
        save_output=ev.save,
        binning=cfg.binning,
    )
    event.completed_time = 1_700_000_000 + t + 0.5
    if ev.is_stim:
        return StimulationData(event, image, pattern, pattern_id, camera_pattern=camera)
    return AcquisitionData(event, image)


# ------------------------------------------------------------- grouping
def test_cadence_groups_common_configuration():
    exp = make_experiment("exp.00", every_t=5, stim_every_t=1)
    exp.stimulation.save = False
    plan = make_plan(make_schedule([exp], steps=12))

    [g] = cadence_groups(plan, "exp.00")
    assert g.name == "imaging"
    assert g.channels == ("545",)
    assert g.every_t == 5
    assert g.timepoints == 3  # t = 0, 5, 10
    assert g.local_index(5) == 1
    assert g.local_index(7) is None
    assert g.global_t(2) == 10


def test_cadence_groups_saved_stimulation_frame():
    same = make_experiment("a.00", every_t=1, stim_every_t=1)
    same.stimulation.save = True
    other = make_experiment("b.00", every_t=5, stim_every_t=1)
    other.stimulation.save = True
    plan = make_plan(make_schedule([same, other], steps=10))

    [g] = cadence_groups(plan, "a.00")
    assert g.channels == ("545", "DMD")  # stimulation frame last in its group
    assert g.stim_channel == "DMD"

    imaging, stim = cadence_groups(plan, "b.00")
    assert (imaging.name, imaging.channels) == ("imaging", ("545",))
    assert (stim.name, stim.channels, stim.every_t) == ("stim", ("DMD",), 1)


# ------------------------------------------------------------- zarr writer
def write_run(
    tmp_path,
    pattern_policy="on_change",
    segmentation=False,
    save_stim=False,
    distinct_ids=False,
):
    exp = make_experiment("exp.00", every_t=2, stim_every_t=1)
    exp.stimulation.save = save_stim
    if segmentation:
        exp.segmentation.save = True
    plan = make_plan(make_schedule([exp], steps=4, interval=3.0))
    core = core_with()
    writer = OMEZarrWriter(pattern_policy=pattern_policy)
    layers = writer.open(plan, core, tmp_path, AFFINE, SLM)

    pats = [np.full(SLM, v, np.uint8) for v in (1, 2, 3)]
    cams = [np.full((16, 16), v, np.uint8) for v in (1, 2, 3)]
    for t in range(4):
        # stimulation every t; the pattern only changes at t=0 and t=2
        # (or every t when distinct_ids, to tell the policies apart)
        pid = f"p{t}" if distinct_ids else ("p0" if t < 2 else "p2")
        writer.write_frame(
            acquire(
                plan,
                "exp.00",
                t,
                "DMD",
                np.full((16, 16), 9, np.uint16),
                pats[0 if t < 2 else 2],
                pid,
                cams[0 if t < 2 else 2],
            )
        )
        if t % 2 == 0:
            writer.write_frame(
                acquire(plan, "exp.00", t, "545", np.full((16, 16), t + 1, np.uint16))
            )
            if segmentation:
                seg_event = acquire(plan, "exp.00", t, "545", None).event
                writer.write_labels(
                    SegmentationData(seg_event, np.full((16, 16), 3, np.uint16))
                )
    writer.close()
    return plan, layers


def test_zarr_layout_and_readback(tmp_path):
    _plan, layers = write_run(tmp_path, segmentation=True)
    root = tmp_path / "exp.00.zarr"
    assert layers == [(str(root.resolve()), "imaging/545")]

    g = zarr.open_group(str(root), mode="r")
    meta = g.attrs["pyclm"]
    assert meta["format"] == 3
    assert meta["ngff_version"] == "0.4"
    assert meta["groups"]["imaging"]["every_t"] == 2
    assert meta["current_t"] == 3
    assert "pyclm" in meta["plan"]

    img = g["imaging/0"]
    assert img.shape == (2, 1, 16, 16)  # compact T: t = 0 and 2
    assert img[0, 0].max() == 1
    assert img[1, 0].max() == 3
    ms = g["imaging"].attrs["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["t", "c", "y", "x"]
    assert (
        ms["datasets"][0]["coordinateTransformations"][0]["scale"][0] == 6.0
    )  # every_t * interval
    assert g["imaging"].attrs["omero"]["channels"][0]["label"] == "545"
    assert g["imaging/labels/segmentation/0"][1, 0].max() == 3

    # pattern saved on change: two distinct ids across four stimulation events,
    # in both spaces
    assert g["patterns/dmd/0"].shape == (2, *SLM)
    assert g["patterns/dmd"].attrs["pyclm"]["pattern_ids"] == ["p0", "p2"]
    assert g["patterns/dmd"].attrs["pyclm"]["camera_ids"] == ["p0", "p2"]
    assert g["patterns/camera/0"].shape == (2, 16, 16)
    assert g["patterns/camera"].attrs["pyclm"]["pattern_ids"] == ["p0", "p2"]

    # frames table: 4 stimulation events + 2 frames, parquet and csv
    assert (tmp_path / "frames.parquet").exists()
    assert (tmp_path / "frames.csv").exists()

    exp = pio.open(root)
    assert exp.format == 3
    assert exp.name == "exp.00"
    grp = exp.groups["imaging"]
    assert grp.acquired() == [0, 1]
    assert grp.frame(1, "545").max() == 3
    assert grp.labels(0, "545").max() == 3
    assert grp.global_t(1) == 2
    assert exp.current_t == 3
    assert exp.pattern_at(1).max() == 1
    assert exp.pattern_at(3).max() == 3
    assert exp.camera_pattern_at(1).max() == 1
    assert exp.camera_pattern_at(3).max() == 3
    assert exp.camera_pattern_at(3).shape == (16, 16)
    assert (
        exp.frames["dmd_index"].to_pylist() == exp.frames["pattern_index"].to_pylist()
    )
    assert exp.frames.num_rows == 6
    assert set(exp.frames["kind"].to_pylist()) == {"stim_event", "frame"}
    assert np.allclose(exp.affine_transform, AFFINE)


def test_zarr_pattern_policy_imaging_and_none(tmp_path):
    # four distinct patterns, imaging frames saved at t = 0 and 2 only
    write_run(tmp_path / "change", pattern_policy="on_change", distinct_ids=True)
    write_run(tmp_path / "imaging", pattern_policy="imaging", distinct_ids=True)
    n = lambda d: zarr.open_array(
        str(tmp_path / d / "exp.00.zarr" / "patterns/dmd/0"), mode="r"
    ).shape[0]
    assert n("change") == 4
    assert n("imaging") == 2
    exp = pio.open(tmp_path / "imaging" / "exp.00.zarr")
    rows = exp.frames.to_pylist()
    stim = {r["t"]: r["pattern_index"] for r in rows if r["kind"] == "stim_event"}
    assert stim == {0: 0, 1: None, 2: 1, 3: None}
    assert exp.camera_pattern_at(1).max() == 1  # the one in force, from t = 0
    write_run(tmp_path / "none", pattern_policy="none")
    assert not (tmp_path / "none" / "exp.00.zarr" / "patterns").exists()
    exp = pio.open(tmp_path / "none" / "exp.00.zarr")
    assert exp.pattern_at(0) is None


def test_skipped_timepoints_cost_no_chunks(tmp_path):
    write_run(tmp_path)
    chunks = sorted(
        p.name
        for p in (tmp_path / "exp.00.zarr" / "imaging" / "0").iterdir()
        if not p.name.startswith(".")
    )
    assert chunks == ["0.0.0.0", "1.0.0.0"]


def test_export_imagej_from_zarr(tmp_path):
    write_run(tmp_path, segmentation=True)
    exp = pio.open(tmp_path / "exp.00.zarr")
    [path] = pio.export_imagej(exp, tmp_path / "export")
    import tifffile

    stack = tifffile.imread(path)
    # T=2 acquired frames; channels = raw, labels, pattern overlay
    assert stack.shape == (2, 3, 16, 16)
    assert stack[1, 0].max() == 3
    assert stack[1, 1].max() == 3
    assert stack[1, 2].max() > 0  # pattern overlay present
    meta = tifffile.TiffFile(path).imagej_metadata
    assert meta["finterval"] == 6.0  # every_t 2 x 3 s interval, in seconds
    assert meta["unit"] == "um"


# ------------------------------------------------------------- hdf5 v1 through the same reader
def test_hdf5_v1_reader_and_export(tmp_path):
    exp_cfg = make_experiment("exp.00", every_t=2, stim_every_t=1)
    exp_cfg.stimulation.save = False
    plan = make_plan(make_schedule([exp_cfg], steps=4))
    core = core_with()
    writer = HDF5WriterV1()
    writer.open(plan, core, tmp_path, AFFINE, SLM)
    for t in range(4):
        writer.write_frame(
            acquire(
                plan,
                "exp.00",
                t,
                "DMD",
                np.zeros((16, 16), np.uint16),
                np.full(SLM, 5, np.uint8),
                "p",
            )
        )
        if t % 2 == 0:
            writer.write_frame(
                acquire(plan, "exp.00", t, "545", np.full((16, 16), t + 1, np.uint16))
            )
    writer.close()

    import h5py

    with h5py.File(tmp_path / "exp.00.hdf5", "r", swmr=True) as f:
        assert f["00000/stim_aq/dmd"].compression == "gzip"  # patterns compress ~50x
        assert f["00000/channel_545/data"].compression is None

    exp = pio.open(tmp_path / "exp.00.hdf5")
    assert exp.format == 1
    g = exp.groups["imaging_545"]
    assert g.every_t == 2
    assert g.acquired() == [0, 1]
    assert g.frame(1, "545").max() == 3
    assert exp.pattern_at(3).max() == 5
    assert exp.current_t == 3
    [path] = pio.export_imagej(exp, tmp_path / "export")
    import tifffile

    assert tifffile.imread(path).shape == (2, 2, 16, 16)  # raw + pattern overlay
    exp.close()


# ------------------------------------------------------ writer process delegates to the writer
def test_writer_process_uses_configured_writer(tmp_path):
    outbox = WriterProcess(
        base_path=tmp_path, stop_event=Event(), writer=make_writer("ome-zarr")
    )
    exp = make_experiment("exp.00")
    plan = make_plan(make_schedule([exp], steps=2))
    outbox.initialize(plan, core_with(), AFFINE, SLM)
    assert outbox.writer.is_open
    outbox.handle_data(acquire(plan, "exp.00", 0, "545", np.ones((16, 16), np.uint16)))
    outbox.close_files()
    assert not outbox.writer.is_open
    assert pio.open(tmp_path / "exp.00.zarr").groups["imaging"].acquired() == [0]


def test_make_writer_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown storage format"):
        make_writer("tiff")


# ------------------------------------------------------------------ tracks
def test_zarr_tracks_layout_readback_and_export(tmp_path):
    from pyclm.core.datatypes import TrackingData
    from pyclm.core.experiments import TrackingConfig
    from pyclm.core.tracking import TrackRow

    exp = make_experiment("exp.00", every_t=1, segmentation_method="cellpose")
    exp.segmentation.save = True
    exp.stimulation.save = False  # one cadence group with the single channel 545
    exp.tracking = TrackingConfig("centroid")
    plan = make_plan(make_schedule([exp], steps=3, interval=2.0))
    writer = OMEZarrWriter()
    recorded = {
        "raw": {("exp.00", "545"), ("exp.00", "DMD")},
        "seg": {("exp.00", "545")},
        "tracks": {("exp.00", "545")},
    }
    routing = {"routes": {"exp.00": {"545": {"tracks": ["pattern@pattern"]}}}}
    writer.open(
        plan, core_with(), tmp_path, AFFINE, SLM, recorded=recorded, routing=routing
    )

    for t in range(3):
        raw = acquire(plan, "exp.00", t, "545", np.full((16, 16), t + 1, np.uint16))
        writer.write_frame(raw)
        labels = np.zeros((16, 16), np.uint16)
        labels[2:6, 2:6] = 1
        labels[10:14, 8 + t : 12 + t] = 2
        writer.write_labels(SegmentationData(raw.event, labels))
        rows = [TrackRow(10, 1, 3.5, 3.5, 16, 0), TrackRow(20, 2, 11.5, 9.5 + t, 16, 0)]
        writer.write_tracks(
            TrackingData(raw.event, labels.astype(np.uint32) * 10, rows)
        )
    writer.close()

    root = tmp_path / "exp.00.zarr"
    g = zarr.open_group(str(root), mode="r")
    assert g["imaging/labels"].attrs["labels"] == ["segmentation", "tracks"]
    assert g["imaging/labels/tracks/0"].dtype == np.uint32
    assert g["imaging/labels/tracks/0"].shape == (3, 1, 16, 16)
    assert g.attrs["pyclm"]["routing"] == routing
    assert (tmp_path / "tracks.parquet").exists()
    assert (tmp_path / "tracks.csv").exists()

    with pio.open(root) as data:
        grp = data.groups["imaging"]
        assert grp.has_labels
        assert grp.has_tracks
        assert grp.tracks(1, "545")[3, 3] == 10
        assert grp.tracks(1, "545")[12, 10] == 20
        assert grp.labels(1, "545")[3, 3] == 1
        table = data.tracks
        assert table.num_rows == 6
        assert table.column("track_id").to_pylist() == [10, 20] * 3
        assert table.column("t").to_pylist() == [0, 0, 1, 1, 2, 2]
        assert table.column("x").to_pylist()[1::2] == [9.5, 10.5, 11.5]
        assert table.column("x_um").to_pylist()[1] == 9.5 * grp.pixel_size_um
        assert data.routing == routing
        [path] = pio.export_imagej(data, tmp_path / "export")

    import tifffile

    stack = tifffile.imread(path)
    assert stack.shape == (3, 3, 16, 16)  # raw, segmentation, tracks (no pattern)
    assert stack[1, 1, 3, 3] == 1
    assert stack[1, 2, 3, 3] == 10


def test_hdf5_v1_drops_tracks_with_one_warning(tmp_path, caplog):
    import logging

    from pyclm.core.datatypes import TrackingData

    exp = make_experiment("exp.00")
    plan = make_plan(make_schedule([exp], steps=2))
    writer = HDF5WriterV1()
    writer.open(plan, core_with(), tmp_path, AFFINE, SLM, routing={"routes": {}})
    raw = acquire(plan, "exp.00", 0, "545", np.ones((16, 16), np.uint16))
    with caplog.at_level(logging.WARNING):
        writer.write_tracks(TrackingData(raw.event, np.zeros((16, 16), np.uint32), []))
        writer.write_tracks(TrackingData(raw.event, np.zeros((16, 16), np.uint32), []))
    writer.close()

    assert caplog.text.count("does not store tracks") == 1
    import h5py

    with h5py.File(tmp_path / "exp.00.hdf5", "r") as f:
        assert json.loads(f.attrs["routing"]) == {"routes": {}}


def test_zarr_named_segmentations_layout_readback_and_export(tmp_path):
    """Two [segmentation] tables of one channel: one label image each, read back by name, exported as two label sets."""
    import tifffile

    from pyclm.core.experiments import SegmentationConfig

    exp = make_experiment("exp.00", every_t=1, segmentation_method="cellpose")
    exp.segmentation.save = True
    exp.segmentations["nuclei"] = SegmentationConfig("cellpose", model="nuclei")
    exp.stimulation.save = False
    plan = make_plan(make_schedule([exp], steps=2, interval=2.0))
    writer = OMEZarrWriter()
    recorded = {
        "raw": {("exp.00", "545"), ("exp.00", "DMD")},
        "seg": {("exp.00", "545")},
        "seg:nuclei": {("exp.00", "545")},
    }
    writer.open(plan, core_with(), tmp_path, AFFINE, SLM, recorded=recorded)
    for t in range(2):
        raw = acquire(plan, "exp.00", t, "545", np.full((16, 16), t + 1, np.uint16))
        writer.write_frame(raw)
        writer.write_frame(
            acquire(plan, "exp.00", t, "DMD", None, np.full(SLM, 1, np.uint8), "p0")
        )
        writer.write_labels(
            SegmentationData(raw.event, np.full((16, 16), 1, np.uint16))
        )
        writer.write_labels(
            SegmentationData(raw.event, np.full((16, 16), 2, np.uint16), "nuclei")
        )
    writer.close()

    import zarr

    g = zarr.open_group(str(tmp_path / "exp.00.zarr"), mode="r")
    assert g["imaging/labels"].attrs["labels"] == ["segmentation", "nuclei"]
    assert g["imaging/labels/nuclei/0"][1, 0].max() == 2

    with pio.open(tmp_path / "exp.00.zarr") as data:
        grp = data.groups["imaging"]
        assert grp.label_names == ("segmentation", "nuclei")
        assert grp.has_labels
        assert grp.labels(0, "545").max() == 1
        assert grp.labels(0, "545", "nuclei").max() == 2
        assert grp.labels(0, "545", "cyto") is None
        paths = pio.export_imagej(data, tmp_path)
    stack = tifffile.imread(paths[0])
    assert stack.shape[1] == 4  # raw, segmentation, nuclei, pattern overlay
    assert stack[0, 2].max() == 2


def test_hdf5_v1_drops_named_segmentations_with_one_warning(tmp_path, caplog):
    import logging

    exp = make_experiment("exp.00", every_t=1, segmentation_method="cellpose")
    plan = make_plan(make_schedule([exp], steps=2))
    writer = HDF5WriterV1()
    writer.open(plan, core_with(), tmp_path)
    raw = acquire(plan, "exp.00", 0, "545", np.zeros((16, 16), np.uint16))
    writer.write_frame(raw)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            writer.write_labels(
                SegmentationData(raw.event, np.ones((16, 16), np.uint16), "nuclei")
            )
        writer.write_labels(SegmentationData(raw.event, np.ones((16, 16), np.uint16)))
    writer.close()
    assert caplog.text.count("stores only the default segmentation") == 1
    with pio.open(tmp_path / "exp.00.hdf5") as data:
        grp = data.groups["imaging_545"]
        assert grp.label_names == ("segmentation",)
        assert grp.labels(0, "545").max() == 1
        assert grp.labels(0, "545", "nuclei") is None
