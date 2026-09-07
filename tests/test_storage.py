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
from pyclm.core.manager import MicroscopeOutbox
from pyclm.core.queues import AllQueues
from pyclm.core.storage import HDF5WriterV1, OMEZarrWriter, cadence_groups, make_writer
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore

AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
SLM = (12, 10)


def core_with(shape=(16, 16)):
    core = SimulatedMicroscopeCore(FakeImageSource(shape), slm_shape=SLM)
    core._roi = (0, 0, shape[1], shape[0])
    core.getROI = lambda: (0, 0, shape[1], shape[0])
    return core


def acquire(plan, name, t, channel, image, pattern=None, pattern_id=None):
    """Build the data object the microscope would deliver for one planned frame."""
    ev = next(
        e
        for e in plan.events_at(t)
        if e.kind == "acquire" and e.experiment == name and e.channel == channel
    )
    cfg = plan.imaging_config(name, channel)
    exp = plan.schedule.experiments[name]
    event = AcquisitionEvent(
        name,
        plan.schedule.positions[name],
        cfg.channel_id,
        index=ev.index,
        scheduled_time=1_700_000_000 + t,
        exposure_time_ms=cfg.exposure,
        needs_slm=ev.is_stim,
        save_output=ev.save,
        segmentation_method=exp.segmentation.method_name,
        pattern_method=exp.pattern.method_name,
        binning=cfg.binning,
    )
    event.completed_time = 1_700_000_000 + t + 0.5
    if ev.is_stim:
        return StimulationData(event, image, pattern, pattern_id)
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
    tmp_path, pattern_policy="on_change", segmentation=False, save_stim=False
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
    for t in range(4):
        # stimulation every t; the pattern only changes at t=0 and t=2
        pid = "p0" if t < 2 else "p2"
        writer.write_frame(
            acquire(
                plan,
                "exp.00",
                t,
                "DMD",
                np.full((16, 16), 9, np.uint16),
                pats[0 if t < 2 else 2],
                pid,
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
    assert meta["format"] == 2
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

    # pattern saved on change: two distinct ids across four stimulation events
    assert g["patterns/dmd/0"].shape == (2, *SLM)
    assert g["patterns/dmd"].attrs["pyclm"]["pattern_ids"] == ["p0", "p2"]

    # frames table: 4 stimulation events + 2 frames, parquet and csv
    assert (tmp_path / "frames.parquet").exists()
    assert (tmp_path / "frames.csv").exists()

    exp = pio.open(root)
    assert exp.format == 2
    assert exp.name == "exp.00"
    grp = exp.groups["imaging"]
    assert grp.acquired() == [0, 1]
    assert grp.frame(1, "545").max() == 3
    assert grp.labels(0, "545").max() == 3
    assert grp.global_t(1) == 2
    assert exp.current_t == 3
    assert exp.pattern_at(1).max() == 1
    assert exp.pattern_at(3).max() == 3
    assert exp.frames.num_rows == 6
    assert set(exp.frames["kind"].to_pylist()) == {"stim_event", "frame"}
    assert np.allclose(exp.affine_transform, AFFINE)


def test_zarr_pattern_policy_all_and_none(tmp_path):
    write_run(tmp_path / "all", pattern_policy="all")
    assert (
        zarr.open_array(
            str(tmp_path / "all" / "exp.00.zarr" / "patterns/dmd/0"), mode="r"
        ).shape[0]
        == 4
    )
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


# ------------------------------------------------------------- outbox delegates to the writer
def test_outbox_uses_configured_writer(tmp_path):
    aq = AllQueues()
    outbox = MicroscopeOutbox(
        aq, base_path=tmp_path, stop_event=Event(), writer=make_writer("ome-zarr")
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
