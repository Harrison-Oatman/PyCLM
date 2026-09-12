"""Grid acquisition through the runtime: SLM buffer, microscope, writer, check, and a dry run."""

import json
import shutil
import threading

import numpy as np
import pytest
import tifffile
import zarr
from helpers import FakeImageSource, make_experiment, make_plan, make_schedule
from test_check import RESOURCES, TOMLS
from test_grid import mm_entry, tile_creator_grid, write_list
from test_storage import AFFINE, SLM, acquire, core_with

import pyclm.io as pio
from pyclm import run_pyclm
from pyclm.check import check_directory
from pyclm.core.datatypes import CameraPattern, EventSLMPattern
from pyclm.core.events import AcquisitionEvent, UpdatePatternEvent
from pyclm.core.experiments import MicroscopePosition
from pyclm.core.grid import GridGeometry, geometry_for
from pyclm.core.manager import SLMBuffer
from pyclm.core.microscope import MicroscopeProcess
from pyclm.core.queues import AllQueues
from pyclm.core.storage import OMEZarrWriter
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore
from pyclm.directories import positions_from_pos

TILES = ((0, 0), (0, 1), (1, 1), (1, 0))


def grid_position(tmp_path, pitch=(10.0, 8.0)):
    """A 2 x 2 grid whose tiles are 10 x 8 px apart at 1 um/px, ROI 10 x 8: no overlap."""
    entries = tile_creator_grid(rows=2, cols=2, pitch=pitch, origin=(0.0, 0.0))
    pos = positions_from_pos(write_list(tmp_path / "PositionList.pos", entries))[0]
    pos.geometry = geometry_for(pos, (0, 0, 10, 8), 1.0)
    return pos


# ------------------------------------------------------------------ SLM buffer
def test_slm_buffer_cuts_a_stitched_pattern_into_tile_images(tmp_path):
    pos = grid_position(tmp_path)
    geom = pos.geometry
    aq = AllQueues()
    slm = SLMBuffer(aq)
    slm.initialize(SLM, AFFINE, ["g.1"], grids={"g.1": geom})
    pid0, blank, cam, ids = slm.slm_patterns["g.1"]
    assert pid0 == 0
    assert len(blank) == 4
    assert cam is None
    assert ids == ["0:0", "0:1", "0:2", "0:3"]

    stitched = np.zeros(geom.shape(1), np.float32)  # (16, 20)
    stitched[:8, 10:] = 1.0  # only tile (0, 1) lit
    slm.handle_data(CameraPattern("g.1", stitched, binning=1))
    pid, dmd, camera, ids = slm.slm_patterns["g.1"]
    assert camera.shape == (16, 20)
    assert camera.max() == 255
    assert [int(d.max()) for d in dmd] == [0, 255, 0, 0]  # tiles in acquisition order
    assert ids == [f"{pid}:{k}" for k in range(4)]

    update = UpdatePatternEvent("g.1")
    from pyclm.core.messages import UpdatePatternEventMessage

    slm.handle_message(UpdatePatternEventMessage(update))
    reply = aq.slm_to_microscope.get_nowait()
    assert isinstance(reply, EventSLMPattern)
    assert reply.dmd_ids == ids
    assert len(reply.pattern) == 4
    assert reply.camera_pattern.shape == (16, 20)


# ------------------------------------------------------------------ microscope
class Sink:
    def __init__(self):
        self.items = []

    def publish(self, data):
        self.items.append(data)
        return 1

    def end_stream(self, name):
        pass


class Recorder(FakeImageSource):
    """Frames whose value is the stage position, so the stitched frame shows the visiting order."""

    def __init__(self, core_ref):
        super().__init__((8, 10))
        self.core_ref = core_ref

    def next_frame(self, pos):
        self.snaps += 1
        value = int(pos[0]) * 10 + int(pos[1])
        return np.full((8, 10), value + 1, np.uint16)


def test_microscope_visits_every_tile_and_publishes_one_stitched_frame(tmp_path):
    pos = grid_position(tmp_path)
    exp = make_experiment("g.1", every_t=1, stim_every_t=1)
    aq = AllQueues()
    holder = {}
    source = Recorder(holder)
    core = SimulatedMicroscopeCore(source)
    core.ROI_FACTOR = 1
    core._roi = (0, 0, 10, 8)
    holder["core"] = core
    microscope = MicroscopeProcess(
        core, aq, stop_event=threading.Event(), settle_time_s=0.0
    )
    microscope.declare_slm()
    microscope.attach(Sink())

    # the handshake delivers one DMD image per tile
    update = UpdatePatternEvent("g.1", index={"t": 0, "p": "g.1", "c": "DMD"})
    tiles = [np.full(SLM, k + 1, np.uint8) for k in range(4)]
    aq.slm_to_microscope.put(
        EventSLMPattern(
            update.id,
            tiles,
            "p",
            camera_pattern=np.ones((16, 20), np.uint8),
            dmd_ids=list("abcd"),
        )
    )
    microscope.handle_update_pattern_event(update, slm_await_s=1.0)
    assert core._slm_image.max() == 1  # the first tile's image is up

    event = AcquisitionEvent(
        "g.1",
        pos,
        exp.stimulation.channel_id,
        index={"t": 0, "p": "g.1", "c": "DMD"},
        exposure_time_ms=1,
        needs_slm=True,
    )
    microscope.handle_acquisition_event(event)

    [data] = microscope.router.items
    assert data.data.shape == (16, 20)  # stitched
    # tile (0,0) at stage (0,0) -> 1; (0,1) at (10,0) -> 101; (1,1) at (10,8) -> 109; (1,0) at (0,8) -> 9
    assert data.data[0, 0] == 1
    assert data.data[0, 19] == 101
    assert data.data[15, 19] == 109
    assert data.data[15, 0] == 9
    assert source.snaps == 4
    assert core._slm_image.max() == 4  # the last tile's image was set before its snap
    assert data.dmd_ids == list("abcd")
    assert len(data.dmd_pattern) == 4
    assert core.getXYPosition() == (0.0, 8.0)  # left at the last tile


# ------------------------------------------------------------------ writer
def test_zarr_writer_stores_stitched_frames_and_one_dmd_image_per_tile(tmp_path):
    pos = grid_position(tmp_path)
    exp = make_experiment("g.1", every_t=1, stim_every_t=1)
    exp.stimulation.save = True
    schedule = make_schedule([exp], steps=2, interval=3.0)
    schedule.positions["g.1"] = pos
    plan = make_plan(schedule)
    writer = OMEZarrWriter()
    writer.open(plan, core_with((8, 10)), tmp_path, AFFINE, SLM)

    stitched = np.full((16, 20), 7, np.uint16)
    tiles = [np.full(SLM, k + 1, np.uint8) for k in range(4)]
    for t in range(2):
        data = acquire(
            plan, "g.1", t, "DMD", stitched, tiles, "p0", np.full((16, 20), 9, np.uint8)
        )
        data.dmd_ids = [f"p0:{k}" for k in range(4)]
        writer.write_frame(data)
        writer.write_frame(acquire(plan, "g.1", t, "545", stitched))
    writer.close()

    root = zarr.open_group(str(tmp_path / "g.1.zarr"), mode="r")
    assert root.attrs["pyclm"]["grid"]["rows"] == 2
    assert root["imaging/0"].shape[2:] == (16, 20)
    assert root["patterns/camera/0"].shape == (1, 16, 20)
    assert root["patterns/dmd/0"].shape == (4, *SLM)
    meta = root["patterns/dmd"].attrs["pyclm"]
    assert meta["pattern_ids"] == ["p0:0", "p0:1", "p0:2", "p0:3"]
    assert meta["camera_ids"] == ["p0"] * 4
    assert meta["tiles"] == [list(rc) for rc in TILES]
    data = pio.open(tmp_path / "g.1.zarr")
    rows = [r for r in data.frames.to_pylist() if r["kind"] == "stim_event"]
    assert [(r["pattern_index"], r["dmd_index"]) for r in rows] == [(0, 0), (0, 0)]
    assert data.grid["columns"] == 2
    assert data.camera_pattern_at(1).shape == (16, 20)


# ------------------------------------------------------------------ check
def test_check_describes_grids_and_refuses_hdf5(tmp_path):
    for name in ("bar10.toml", "pyclm_config.toml"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    entries = [
        mm_entry("bar10-1", 0.0, 0.0, 10.0, 0, 0),
        *tile_creator_grid(prefix="bar10-2", pitch=(600.0, 600.0)),
    ]
    write_list(tmp_path / "PositionList.pos", entries)

    report = check_directory(tmp_path, pixel_size_um=1.0)
    text = report.text()
    assert "bar10.2: grid of 2 x 3 tiles, spacing 600.0 x 600.0 um" in text
    assert "set camera_roi to check" in text
    assert any('need format = "ome-zarr"' in str(e) for e in report.errors)

    cfg = (tmp_path / "pyclm_config.toml").read_text()
    cfg = cfg.replace('format = "hdf5"', 'format = "ome-zarr"').replace(
        "config_path", "camera_roi = [0, 0, 500, 500]\nconfig_path", 1
    )
    (tmp_path / "pyclm_config.toml").write_text(cfg)
    report = check_directory(tmp_path, pixel_size_um=1.0)
    assert report.ok or not report.errors, report.text()
    assert any("spaced wider than the camera ROI" in str(w) for w in report.warnings)


# ------------------------------------------------------------------ end to end
def test_dry_run_grid(tmp_path):
    """
    A 2 x 2 grid on the virtual microscope: one TIF the size of the grid, a
    camera ROI of one tile; the run stitches the tiles back into that image,
    stores one DMD image per tile, and the pattern method sees the whole grid.
    """
    for name in ("bar10.toml", "pyclm_config.toml"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    cfg = (tmp_path / "pyclm_config.toml").read_text()
    cfg = cfg.replace('format = "hdf5"', 'format = "ome-zarr"').replace(
        "config_path", "camera_roi = [0, 0, 800, 800]\nconfig_path", 1
    )
    (tmp_path / "pyclm_config.toml").write_text(cfg)
    toml = tmp_path / "bar10.toml"
    text = toml.read_text()
    toml.write_text(text[: text.index("[pattern]")] + '[pattern]\nmethod = "full_on"\n')

    # tiles 800 um apart at 1 um/px: 800 reported px = 200 TIF px (ROI factor 4)
    entries = tile_creator_grid(
        prefix="bar10-1", rows=2, cols=2, pitch=(800.0, 800.0), origin=(0.0, 0.0)
    )
    write_list(tmp_path / "PositionList.pos", entries)
    image = np.zeros((400, 400), np.uint16)
    image[:200, :200], image[:200, 200:], image[200:, :200], image[200:, 200:] = (
        1,
        2,
        3,
        4,
    )
    tifffile.imwrite(tmp_path / "bar10.1.tif", image)
    (tmp_path / "dry_run.yml").write_text("pixel_size_um: 1.0\n")

    run_pyclm(tmp_path, dry=True)

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["done"] is True
    with pio.open(tmp_path / "bar10.1.zarr") as exp:
        assert exp.grid["rows"] == 2
        assert exp.grid["columns"] == 2
        frame = exp.groups["imaging"].frame(0, "545")
        assert frame.shape == (400, 400)
        assert frame[0, 0] == 1
        assert frame[0, 399] == 2
        assert frame[399, 0] == 3
        assert frame[399, 399] == 4
        ids = exp.pattern_ids
        assert len(ids) % 4 == 0
        assert ids[0] == "0:0"
        rows = [
            r
            for r in exp.frames.to_pylist()
            if r["kind"] == "frame" and r["channel"] == "545"
        ]
        assert all(r["x"] == 400.0 and r["y"] == 400.0 for r in rows)  # the grid centre
        # the pattern covers the whole grid; on the virtual microscope it is at
        # the reported camera scale (ROI_FACTOR times the TIF), as for any dry run
        cam = exp.camera_pattern_at(3)
        assert cam is not None
        assert cam.shape == (1600, 1600)
        assert cam.min() == 255
