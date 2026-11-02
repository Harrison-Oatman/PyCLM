"""
The [acquisition] options of pyclm_config.toml, against the simulated core:
settle only after a stage move, a one-image sequence acquisition instead of
snapImage, a stimulation timed by the shutter when nothing keeps or reads
its frame, and exposure / binning set only when the camera has other values.
"""

import logging
from threading import Event
from time import perf_counter

import numpy as np
import pytest
from helpers import FakeImageSource, make_experiment
from test_dry_run import yml_experiment_dir

from pyclm.core.datatypes import StimulationData
from pyclm.core.events import AcquisitionEvent, UpdateStagePositionEvent
from pyclm.core.experiments import MicroscopePosition
from pyclm.core.microscope import MicroscopeProcess
from pyclm.core.queues import AllQueues
from pyclm.core.virtual_microscope.simulated_core import SimulatedMicroscopeCore
from pyclm.schema import ConfigError, PyclmConfig, validate_model

EXP = make_experiment("exp.00")
POSITION = MicroscopePosition(0.0, 0.0, 0.0, label="exp.00")


class Sink:
    def __init__(self, demanded=()):
        self.items = []
        self._demanded = set(demanded)

    def publish(self, data):
        self.items.append(data)
        return 1

    def demanded(self, kind):
        return self._demanded if kind == "raw" else set()

    def end_stream(self, name):
        pass


class CountingCore(SimulatedMicroscopeCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def snapImage(self):
        self.calls.append("snap")
        super().snapImage()

    def startSequenceAcquisition(self, n, interval_ms, stop_on_overflow):
        self.calls.append("sequence")
        super().startSequenceAcquisition(n, interval_ms, stop_on_overflow)

    def setExposure(self, exposure_ms):
        self.calls.append(("exposure", exposure_ms))
        super().setExposure(exposure_ms)

    def setProperty(self, label, name, value):
        if name == "Binning":
            self.calls.append(("binning", value))
        super().setProperty(label, name, value)


def make_microscope(core=None, router=None, **kwargs):
    core = core or CountingCore(FakeImageSource((16, 16)))
    microscope = MicroscopeProcess(
        core, AllQueues(), stop_event=Event(), settle_time_s=0.0, **kwargs
    )
    microscope.declare_slm()
    microscope.attach(router or Sink())
    return microscope, core


def event(t=0, channel="545", needs_slm=False, save=True, exposure_ms=5, binning=1):
    cfg = EXP.stimulation if channel == "stimulation" else EXP.channels[channel]
    return AcquisitionEvent(
        EXP.experiment_name,
        POSITION,
        cfg.channel_id,
        index={"t": t, "p": "exp.00", "c": channel},
        exposure_time_ms=exposure_ms,
        needs_slm=needs_slm,
        save_output=save,
        binning=binning,
    )


def stim(t=0, save=False, exposure_ms=50):
    return event(t, "stimulation", needs_slm=True, save=save, exposure_ms=exposure_ms)


# ------------------------------------------------------------------ settle
@pytest.mark.parametrize(
    ("mode", "second_waits"), [("every_frame", True), ("after_move", False)]
)
def test_settle_after_a_move_only(mode, second_waits):
    microscope, _core = make_microscope(settle=mode)
    microscope.settle_time_s = 0.2
    microscope.handle_update_position_event(
        UpdateStagePositionEvent(POSITION, EXP.experiment_name)
    )

    start = perf_counter()
    microscope.handle_acquisition_event(event(channel="545"))
    first = perf_counter() - start
    start = perf_counter()
    microscope.handle_acquisition_event(event(channel="545"))
    second = perf_counter() - start

    assert first >= 0.2
    assert (second >= 0.2) is second_waits


def test_unknown_modes_are_refused():
    with pytest.raises(ValueError, match="settle"):
        make_microscope(settle="sometimes")
    with pytest.raises(ValueError, match="camera_capture"):
        make_microscope(camera_capture="live")
    with pytest.raises(ValueError, match="stimulation_timing"):
        make_microscope(stimulation_timing="dmd")


# ---------------------------------------------------------------- sequence
def test_sequence_capture_takes_the_frame_from_a_one_image_sequence():
    microscope, core = make_microscope(camera_capture="sequence")
    microscope.handle_acquisition_event(event())
    frame = microscope.router.items[-1]
    assert frame.data.shape == (16, 16)
    assert core.calls.count("sequence") == 1
    assert core.getRemainingImageCount() == 0


def test_a_sequence_that_gives_no_image_times_out_and_is_stopped():
    class Stuck(CountingCore):
        stopped = 0

        def startSequenceAcquisition(self, n, interval_ms, stop_on_overflow):
            self.running = True

        def isSequenceRunning(self):
            return getattr(self, "running", False)

        def stopSequenceAcquisition(self):
            self.running = False
            Stuck.stopped += 1

    microscope, core = make_microscope(
        Stuck(FakeImageSource()), camera_capture="sequence"
    )
    microscope.sequence_timeout_s = 0.05
    with pytest.raises(TimeoutError, match="no image"):
        microscope.snap()
    assert Stuck.stopped == 1
    assert not core.isSequenceRunning()


# ----------------------------------------------------------------- shutter
def test_a_stimulation_nobody_keeps_is_timed_by_the_shutter():
    microscope, core = make_microscope(stimulation_timing="shutter")
    start = perf_counter()
    microscope.handle_acquisition_event(stim(exposure_ms=50))
    elapsed = perf_counter() - start

    out = microscope.router.items[-1]
    assert isinstance(out, StimulationData)
    assert out.data is None
    assert "snap" not in core.calls
    assert core.shutter_openings == 1
    assert not core.shutter_open
    assert elapsed >= 0.05
    assert microscope.shutter_exposures == 1
    assert "1 stimulation(s) by shutter" in microscope.summary()


@pytest.mark.parametrize(
    ("save", "demanded", "timing"),
    [
        (True, (), "shutter"),  # the frame is saved
        (False, {("exp.00", "stimulation")}, "shutter"),  # a method reads it
        (False, (), "camera"),  # the default
    ],
)
def test_the_camera_still_times_a_stimulation_whose_frame_is_used(
    save, demanded, timing
):
    microscope, core = make_microscope(router=Sink(demanded), stimulation_timing=timing)
    microscope.handle_acquisition_event(stim(save=save, exposure_ms=1))
    assert core.calls.count("snap") == 1
    assert core.shutter_openings == 0
    assert microscope.router.items[-1].data is not None


def test_no_shutter_device_falls_back_to_the_camera(caplog):
    core = CountingCore(FakeImageSource())
    core._shutter_device = ""
    microscope, _ = make_microscope(core, stimulation_timing="shutter")
    with caplog.at_level(logging.WARNING):
        microscope.handle_acquisition_event(stim(exposure_ms=1))
        microscope.handle_acquisition_event(stim(t=1, exposure_ms=1))
    assert core.calls.count("snap") == 2
    assert caplog.text.count("no current shutter") == 1


# ----------------------------------------------------------- camera values
@pytest.mark.parametrize("reapply", [False, True])
def test_exposure_and_binning_are_set_only_when_they_differ(reapply):
    microscope, core = make_microscope(reapply_camera_settings=reapply)
    microscope.handle_acquisition_event(event(exposure_ms=20, binning=2))
    microscope.handle_acquisition_event(event(exposure_ms=20, binning=2))
    microscope.handle_acquisition_event(event(exposure_ms=30, binning=1))
    sets = [c for c in core.calls if isinstance(c, tuple)]
    if reapply:
        assert len(sets) == 6
    else:
        assert sets == [
            ("exposure", 20.0),
            ("binning", "2x2"),
            ("exposure", 30.0),
            ("binning", "1x1"),
        ]
        assert microscope.camera_sets_skipped == 2
    # the camera's own value decides, not a memory: a change made behind
    # PyCLM's back is set again
    core.calls.clear()
    core._exposure_ms = 99.0
    microscope.handle_acquisition_event(event(exposure_ms=30, binning=1))
    assert ("exposure", 30.0) in core.calls


# ------------------------------------------------------------------ schema
def test_acquisition_table_in_pyclm_config():
    base = {
        "config_path": "x.cfg",
        "affine_transform": [[1, 0, 0], [0, 1, 0]],
        "slm_shape_h": 10,
        "slm_shape_w": 10,
    }
    cfg = validate_model(PyclmConfig, base, None)
    assert cfg.acquisition.model_dump() == {
        "settle": "every_frame",
        "camera_capture": "snap",
        "stimulation_timing": "camera",
        "reapply_camera_settings": False,
    }
    cfg = validate_model(
        PyclmConfig,
        {**base, "acquisition": {"settle": "after_move", "camera_capture": "sequence"}},
        None,
    )
    assert cfg.acquisition.settle == "after_move"
    with pytest.raises(ConfigError, match="stimulation_timing"):
        validate_model(
            PyclmConfig, {**base, "acquisition": {"stimulation_timing": "dmd"}}, None
        )
    with pytest.raises(ConfigError, match="settle_s"):
        validate_model(PyclmConfig, {**base, "acquisition": {"settle_s": 1}}, None)


def test_grid_stimulation_by_shutter_gives_no_frame():
    from pyclm.core.grid import GridGeometry

    microscope, core = make_microscope(stimulation_timing="shutter")
    tiles = [MicroscopePosition(0.0, 0.0, 0.0), MicroscopePosition(10.0, 0.0, 0.0)]
    pos = MicroscopePosition(0.0, 0.0, 0.0, label="exp.00")
    pos.tiles = tiles
    pos.geometry = GridGeometry(1, 2, 16.0, 16.0, 16, 16, ((0, 0), (0, 1)))
    ev = stim(exposure_ms=1)
    ev.position = pos
    microscope.current_pattern = [np.zeros((4, 4)), np.zeros((4, 4))]
    microscope.handle_acquisition_event(ev)
    assert microscope.router.items[-1].data is None
    assert core.shutter_openings == 2


# ----------------------------------------------------------------- dry run
def test_dry_run_with_every_option_on(yml_experiment_dir, caplog):
    import json

    import pyclm.io as pio
    from pyclm import run_pyclm

    d = yml_experiment_dir
    config = d / "pyclm_config.toml"
    config.write_text(
        config.read_text().replace('format = "hdf5"', 'format = "ome-zarr"')
        + '\n[acquisition]\nsettle = "after_move"\ncamera_capture = "sequence"\n'
        'stimulation_timing = "shutter"\n'
    )
    for name in ("bar10", "bar025"):
        toml = d / f"{name}.toml"
        text = toml.read_text()
        i = text.index("[stimulation]")
        toml.write_text(text[:i] + text[i:].replace("save = true", "save = false", 1))

    with caplog.at_level(logging.INFO, logger="pyclm.core.microscope"):
        run_pyclm(d, dry=True)
    summary = [r.message for r in caplog.records if "by shutter" in r.message]
    assert summary
    assert "settle after_move, camera sequence" in summary[0]
    assert " 0 stimulation(s) by shutter" not in summary[0]

    assert json.loads((d / "status.json").read_text())["done"] is True
    for store in sorted(d.glob("*.zarr")):
        with pio.open(store) as exp:
            rows = exp.frames.to_pylist()
            stim = [r for r in rows if r["channel"] == "DMD"]
            assert stim
            assert {r["kind"] for r in stim} <= {"stim_event", "skipped"}
            assert any(r["kind"] == "frame" and r["channel"] == "545" for r in rows)


def test_check_says_when_the_camera_still_times_stimulations(yml_experiment_dir):
    from pyclm.check import check_directory

    d = yml_experiment_dir
    config = d / "pyclm_config.toml"
    config.write_text(
        config.read_text() + '\n[acquisition]\nstimulation_timing = "shutter"\n'
    )
    report = check_directory(d)
    notes = [str(f) for f in report.findings if "timed by the camera" in str(f)]
    assert len(notes) == 2  # bar10 and bar025 both save their stimulation frames
    assert "save = true" in notes[0]
