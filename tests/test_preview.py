"""
pyclm preview: an open-loop method on a TIF, a closed-loop method with
segmentation and tracking, and the outputs a user looks at.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pyclm.core.patterns import PatternMethod
from pyclm.core.segmentation import SegmentationMethod
from pyclm.preview import parse_timepoints, preview
from pyclm.schema import ConfigError

RESOURCES = Path(__file__).parent / "dry_run_resources"
TOMLS = RESOURCES / "tomls"


@pytest.fixture
def exp_dir(tmp_path):
    for name in ("bar10.toml", "pyclm_config.toml"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    return tmp_path


@pytest.fixture
def tif(tmp_path):
    rng = np.random.default_rng(0)
    frame = rng.integers(100, 200, size=(120, 160), dtype=np.uint16)
    frame[20:40, 30:50] = 4000
    frame[70:90, 100:130] = 3000
    path = tmp_path / "frame.tif"
    tifffile.imwrite(path, frame)
    return path


def test_open_loop_preview_writes_pattern_and_overlay(exp_dir, tif):
    result = preview(exp_dir, "bar10", image=tif, t=2, pixel_size_um=0.5)

    assert result.label == "bar10.preview"
    assert result.out_dir == exp_dir / "preview" / "bar10.preview"
    assert set(result.paths) == {
        "raw 545",
        "pattern (camera)",
        "overlay",
        "pattern (DMD)",
        "summary",
    }
    assert result.pattern.shape == (120, 160)
    assert 0 < result.pattern.mean() < 1
    assert tifffile.imread(result.paths["pattern (camera)"]).shape == (120, 160)
    assert tifffile.imread(result.paths["pattern (DMD)"]).shape == (1140, 912)
    info = json.loads(result.paths["summary"].read_text())
    assert info["method"] == "bar"
    assert info["kwargs"]["speed_um_per_min"] == 1.0
    assert info["t"] == 2
    assert info["time_s"] == 2.0
    assert info["pixel_size_um"] == 0.5
    assert info["requirements"] == {}
    assert 0 < info["lit_fraction"] < 1
    assert "bar10.preview" in result.text()


class Threshold(SegmentationMethod):
    name = "threshold"

    def segment(self, data):
        from skimage.measure import label

        return label(data > 1000).astype(np.uint16)


class Follow(PatternMethod):
    name = "follow"

    def __init__(self, channel="545", **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.add_requirement(channel, raw=True, seg=True, tracks=True)

    def generate(self, context):
        tracks = context.tracks(self.channel)
        context.set_exposure(self.channel, 42)
        return tracks.paint(1.0)


def test_closed_loop_preview_runs_segmentation_and_tracking(exp_dir, tif):
    toml = exp_dir / "bar10.toml"
    text = toml.read_text()
    toml.write_text(
        text[: text.index("[pattern]")]
        + '[segmentation]\nmethod = "threshold"\n\n[tracking]\nmethod = "centroid"\n\n'
        '[pattern]\nmethod = "follow"\nchannel = "545"\n'
    )
    result = preview(
        exp_dir,
        "bar10.00",
        image=tif,
        segmentation_methods={"threshold": Threshold},
        pattern_methods={"follow": Follow},
    )

    assert result.label == "bar10.00"
    assert "labels 545 segmentation" in result.paths
    assert "labels 545 tracks" in result.paths
    labels = tifffile.imread(result.paths["labels 545 segmentation"])
    assert labels.max() == 2
    tracks = tifffile.imread(result.paths["labels 545 tracks"])
    assert tracks.dtype == np.uint32
    assert set(np.unique(tracks)) == {0, 1, 2}
    # the pattern is the tracked cells
    assert np.all(result.pattern[labels > 0] == 1.0)
    assert result.pattern[labels == 0].sum() == 0
    assert result.requests == ["exposure 545.exposure_ms = 42.0"]
    info = json.loads(result.paths["summary"].read_text())
    assert info["objects"] == {"segmentation": 2}
    assert info["requirements"] == {"545": ["raw", "seg", "tracks"]}
    assert "would change: exposure 545.exposure_ms = 42.0" in result.text()


def test_preview_errors(exp_dir, tif):
    with pytest.raises(ConfigError, match="no experiment file"):
        preview(exp_dir, "nope", image=tif)
    with pytest.raises(ValueError, match="needs an image"):
        preview(exp_dir, "bar10")


def test_stimulation_only_preview(exp_dir, tif):
    """No [channels]: the probe is the stimulation frame and the method still runs."""
    (exp_dir / "stim.toml").write_text(
        "format_version = 1\n[stimulation]\nexposure = 100\n\n"
        '    [stimulation.config_groups]\n    Channel = "DMD"\n\n'
        '[pattern]\nmethod = "full_on"\n'
    )
    result = preview(exp_dir, "stim", image=tif, pixel_size_um=0.5)
    assert "raw DMD" in result.paths
    assert result.pattern.shape == (120, 160)
    assert result.pattern.min() == 1.0


def test_series_preview_writes_stacks_montage_and_animation(exp_dir, tif):
    """Several timepoints: the bar moves, and every pattern file is a stack."""
    schedule = exp_dir / "schedule.toml"
    schedule.write_text(
        schedule.read_text().replace(
            "interval_seconds = 1.0", "interval_seconds = 60.0"
        )
    )
    result = preview(exp_dir, "bar10", image=tif, t=range(0, 61, 10), pixel_size_um=0.5)

    camera = tifffile.imread(result.paths["pattern (camera)"])
    assert camera.shape == (7, 120, 160)
    assert tifffile.imread(result.paths["pattern (DMD)"]).shape == (7, 1140, 912)
    # a moving bar: no two timepoints light the same pixels
    assert all(not np.array_equal(camera[i], camera[i + 1]) for i in range(6))
    assert np.array_equal(camera[-1], np.clip(result.pattern, 0, 1))
    assert result.paths["montage"].exists()
    assert result.paths["animation"].exists()
    info = json.loads(result.paths["summary"].read_text())
    assert [row["t"] for row in info["timepoints"]] == [0, 10, 20, 30, 40, 50, 60]
    assert all(row["active"] and row["stimulated"] for row in info["timepoints"])
    assert info["timepoints"][1]["time_s"] == 600.0
    assert info["t"] == 60
    assert "7 timepoints, t = 0 to 60" in result.text()


def test_series_preview_matches_single_timepoints(exp_dir, tif):
    """Each frame of a series is what a one-timepoint preview gives at that t."""
    series = preview(exp_dir, "bar10", image=tif, t=[3, 7], pixel_size_um=0.5)
    stack = tifffile.imread(series.paths["pattern (camera)"])
    for i, t in enumerate((3, 7)):
        single = preview(exp_dir, "bar10", image=tif, t=t, pixel_size_um=0.5)
        assert np.array_equal(stack[i], np.clip(single.pattern, 0, 1))


def test_series_preview_is_dark_outside_t_delay_and_t_stop(exp_dir, tif):
    toml = exp_dir / "bar10.toml"
    toml.write_text("t_delay = 2\nt_stop = 3\n" + toml.read_text())
    result = preview(exp_dir, "bar10", image=tif, t=range(0, 7), pixel_size_um=0.5)

    camera = tifffile.imread(result.paths["pattern (camera)"])
    info = json.loads(result.paths["summary"].read_text())
    active = [row["active"] for row in info["timepoints"]]
    assert active == [False, False, True, True, True, False, False]
    for i, on in enumerate(active):
        assert (camera[i].max() > 0) == on


def test_series_preview_keeps_state_between_timepoints(exp_dir, tif):
    """A closed loop sees its own previous pattern and the tracker keeps its ids."""
    toml = exp_dir / "bar10.toml"
    text = toml.read_text()
    toml.write_text(
        text[: text.index("[pattern]")]
        + '[segmentation]\nmethod = "threshold"\n\n[tracking]\nmethod = "centroid"\n\n'
        '[pattern]\nmethod = "alternate"\n'
    )

    class Alternate(PatternMethod):
        """Light the tracked cells at every other generation."""

        name = "alternate"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_requirement("545", raw=True, seg=True, tracks=True)

        def generate(self, context):
            on = context.last_pattern() is None or context.last_pattern().max() == 0
            return context.tracks("545").paint(1.0 if on else 0.0)

    result = preview(
        exp_dir,
        "bar10.00",
        image=tif,
        t=range(4),
        segmentation_methods={"threshold": Threshold},
        pattern_methods={"alternate": Alternate},
    )
    camera = tifffile.imread(result.paths["pattern (camera)"])
    assert [bool(frame.max()) for frame in camera] == [True, False, True, False]
    tracks = tifffile.imread(result.paths["labels 545 tracks"])
    assert set(np.unique(tracks)) == {0, 1, 2}


def test_parse_timepoints():
    assert parse_timepoints("5") == [5]
    assert parse_timepoints("0:10:5") == [0, 5, 10]
    assert parse_timepoints("2:4") == [2, 3, 4]
    for bad in ("a", "5:1", "0:10:0", "1:2:3:4", "-1:3"):
        with pytest.raises(ValueError, match="start:stop"):
            parse_timepoints(bad)
