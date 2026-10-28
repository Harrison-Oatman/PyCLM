"""
Multi-channel segmentation: a [segmentation] table's ``channels`` feeds the
method one (height, width, channel) image per timepoint, and Cellpose's
normalization (fixed bounds per channel, or each frame's own percentiles).
"""

from typing import ClassVar

import numpy as np
import pytest
from helpers import drain, make_plan, make_schedule

from pyclm.core.datatypes import AcquisitionData
from pyclm.core.events import AcquisitionEvent
from pyclm.core.pattern_process import PatternProcess
from pyclm.core.patterns import AcquiredImageRequest
from pyclm.core.queues import AllQueues
from pyclm.core.router import Router
from pyclm.core.segmentation import SegmentationMethod
from pyclm.core.segmentation.cellpose_segmentation import (
    CellposeSegmentationMethod,
    normalization,
)
from pyclm.core.segmentation_process import SegmentationProcess
from pyclm.core.writer_process import WriterProcess
from pyclm.schema import ConfigError, ExperimentConfig, validate_model

BASE = {
    "imaging": {"exposure": 10, "every_t": 2},
    "channels": {"group": "Channel", "presets": ["545", "638"]},
    "stimulation": {"exposure": 10},
    "pattern": {"method": "full_on"},
}


def config(**segmentation):
    data = dict(BASE)
    data["segmentation"] = {"method": "two_color", **segmentation}
    return validate_model(ExperimentConfig, data)


class TwoColor(SegmentationMethod):
    """Labels pixels lit in both channels; records what it was given."""

    name = "two_color"
    max_channels = 3
    seen: ClassVar[list] = []

    def segment(self, data):
        TwoColor.seen.append(np.asarray(data).shape)
        data = np.asarray(data)
        return (data.min(axis=-1) > 0).astype(np.uint16)


class Camera:
    name = "microscope"
    produces: ClassVar[dict] = {"raw": ()}
    always_active = True

    def attach(self, router, inbox):
        pass


def event_for(exp, t, channel):
    return AcquisitionEvent(
        exp.experiment_name,
        None,
        exp.channels[channel].channel_id,
        index={"t": t, "p": exp.experiment_name, "c": channel},
    )


# ------------------------------------------------------------------ schema
def test_channels_must_be_imaging_channels_at_one_cadence():
    assert config(channels=["545", "638"]).segmentation.channels == ["545", "638"]
    with pytest.raises(ConfigError, match="not imaging channels"):
        config(channels=["545", "GFP"])
    with pytest.raises(ConfigError, match="twice"):
        config(channels=["545", "545"])
    data = dict(BASE)
    data["channels"] = {
        "group": "Channel",
        "presets": ["545", "638"],
        "638": {"every_t": 4},
    }
    data["segmentation"] = {"method": "two_color", "channels": ["545", "638"]}
    with pytest.raises(ConfigError, match="different every_t"):
        validate_model(ExperimentConfig, data)
    # channels is the table's own key, not a method argument
    seg = config(channels=["545", "638"]).to_experiment("e.00").segmentation
    assert seg.channels == ["545", "638"]
    assert "channels" not in seg.kwargs


# ---------------------------------------------------------- routing and run
def test_the_method_gets_both_channels_of_one_timepoint_as_one_image():
    exp = config(channels=["545", "638"]).to_experiment("exp.00")
    cid = exp.channels["545"].channel_id
    reqs = [AcquiredImageRequest(cid, False, True)]
    plan = make_plan(make_schedule([exp], steps=4), {"exp.00": reqs})

    seg = SegmentationProcess()
    seg.register_method(TwoColor, "two_color")
    router = Router(plan)
    for proc in (
        Camera(),
        WriterProcess(base_path=None),
        seg,
        PatternProcess(AllQueues()),
    ):
        router.add(proc)
    router.resolve()

    routes = router.as_dict()["routes"]["exp.00"]
    # the segmentation of 545 reads 638's frames too
    assert routes["638"]["raw"] == ["segmentation@pattern", "writer(record)"]
    assert routes["545"]["raw"] == ["segmentation@pattern", "writer(record)"]
    # and nothing segments 638 itself
    assert router.produced_by("segmentation") == {
        ("exp.00", "545"): [("seg", "pattern")]
    }
    assert routes["545"]["seg"][0].startswith("pattern")

    seg.request_method(exp)
    TwoColor.seen.clear()
    a = np.zeros((6, 8), np.uint16)
    a[:, :4] = 5
    b = np.zeros((6, 8), np.uint16)
    b[:3, :] = 7
    seg.handle_data(AcquisitionData(event_for(exp, 0, "545"), a))
    assert drain(router.inbox("pattern")) == []  # waits for 638
    seg.handle_data(AcquisitionData(event_for(exp, 0, "638"), b))
    out = [d for d in drain(router.inbox("pattern")) if d.kind == "seg"]
    assert len(out) == 1
    assert TwoColor.seen == [(6, 8, 2)]
    assert out[0].event.index["c"] == "545"
    assert out[0].data.sum() == 3 * 4  # lit in both

    # a timepoint that never gets its second channel is dropped, with a warning
    seg.handle_data(AcquisitionData(event_for(exp, 2, "545"), a))
    seg.handle_data(AcquisitionData(event_for(exp, 4, "545"), a))
    seg.handle_data(AcquisitionData(event_for(exp, 4, "638"), b))
    assert [
        d.event.t_index for d in drain(router.inbox("pattern")) if d.kind == "seg"
    ] == [4]
    assert not seg._pending


def test_frames_of_different_shapes_are_refused():
    from pyclm.core.segmentation_process import stack_channels

    exp = config(channels=["545", "638"]).to_experiment("exp.00")
    frames = [
        AcquisitionData(event_for(exp, 0, "545"), np.zeros((4, 4))),
        AcquisitionData(event_for(exp, 0, "638"), np.zeros((4, 5))),
    ]
    with pytest.raises(ValueError, match="same binning"):
        stack_channels(frames)


# ------------------------------------------------------------------- check
def _check_dir(tmp_path, segmentation: str, pattern: str = 'method = "full_on"'):
    import shutil

    from test_check import RESOURCES, TOMLS

    for name in ("pyclm_config.toml", "PositionList.pos"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    text = (TOMLS / "bar10.toml").read_text()
    text = text.replace('presets = ["545"]', 'presets = ["545", "638"]')
    text = text.replace("[pattern]", segmentation + "\n\n[pattern]")
    (tmp_path / "bar10.toml").write_text(text)
    shutil.copy(TOMLS / "bar025.toml", tmp_path / "bar025.toml")
    return tmp_path


def test_check_refuses_more_channels_than_the_method_takes(tmp_path):
    from pyclm.check import check_directory

    d = _check_dir(
        tmp_path,
        '[segmentation]\nmethod = "cellpose"\nchannels = ["545", "638"]',
    )
    report = check_directory(d)
    assert not any("channels" in f.where for f in report.errors), report.text()

    class OneChannel(SegmentationMethod):
        name = "one_channel"

    (d / "bar10.toml").write_text(
        (d / "bar10.toml").read_text().replace('"cellpose"', '"one_channel"')
    )
    report = check_directory(d, segmentation_methods={"one_channel": OneChannel})
    assert any("takes at most 1 channel" in str(f) for f in report.errors), (
        report.text()
    )


# ---------------------------------------------------------------- cellpose
def test_cellpose_normalization_fixed_per_channel_and_auto():
    assert normalization(False, 0, 5000, 1, 99) == {"lowhigh": [0, 5000]}
    assert normalization(True, 0, 5000, 2, 98) == {
        "normalize": True,
        "percentile": [2.0, 98.0],
    }
    per_channel = normalization(False, [100, 200], [4000, 900], 1, 99, n_channels=2)
    # padded to the three channels Cellpose works in
    assert per_channel == {"lowhigh": [[100.0, 4000.0], [200.0, 900.0], [0.0, 1.0]]}
    one_pair = normalization(False, 0, [4000, 900], 1, 99, n_channels=2)
    assert one_pair["lowhigh"][:2] == [[0.0, 4000.0], [0.0, 900.0]]
    with pytest.raises(ValueError, match="one per listed channel"):
        normalization(False, [1, 2, 3], 10, 1, 99, n_channels=2)


def test_cellpose_passes_a_color_image_and_its_normalization():
    class FakeResource:
        def __init__(self):
            self.calls = []

        def eval(self, data, **kwargs):
            self.calls.append((np.asarray(data).shape, kwargs["normalize"]))
            return np.zeros(np.asarray(data).shape[:2], np.uint16)

    method = CellposeSegmentationMethod(
        "e.00", auto_normalize=True, percentile_high=99.5
    )
    method.provide_resource(FakeResource())
    method.segment(np.zeros((8, 8, 2)))
    method.segment(np.zeros((8, 8)))
    calls = method.cellpose_resource.calls
    assert calls[0] == ((8, 8, 2), {"normalize": True, "percentile": [1.0, 99.5]})
    assert calls[1][0] == (8, 8)

    fixed = CellposeSegmentationMethod("e.00", normlow=[0, 10], normhigh=[100, 200])
    fixed.provide_resource(FakeResource())
    fixed.segment(np.zeros((8, 8, 2)))
    assert fixed.cellpose_resource.calls[0][1]["lowhigh"][:2] == [
        [0.0, 100.0],
        [10.0, 200.0],
    ]
    with pytest.raises(ValueError, match="percentile"):
        CellposeSegmentationMethod("e.00", percentile_low=50, percentile_high=10)
