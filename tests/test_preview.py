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
from pyclm.preview import preview
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
    assert info["kwargs"]["bar_speed"] == 1.0
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
