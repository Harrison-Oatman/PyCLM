import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, Union
from uuid import UUID, uuid4

import numpy as np
from h5py import File
from natsort import natsorted

from ..datatypes import AcquisitionData, SegmentationData
from ..experiments import Experiment

logger = logging.getLogger(__name__)


class ROI(NamedTuple):
    x_offset: int
    y_offset: int
    width: int
    height: int


class CameraProperties(NamedTuple):
    roi: ROI
    pixel_size_um: float


class AcquiredImageRequest(NamedTuple):
    id: UUID
    needs_raw: bool
    needs_seg: bool


class DataDock:
    def __init__(self, time_seconds, requirements: list[AcquiredImageRequest]):
        self.time_seconds = time_seconds
        self.requirements = requirements
        self.data = defaultdict(dict)

        for channel_id, needs_raw, needs_seg in requirements:
            if needs_raw:
                self.data[channel_id]["raw"] = None

            if needs_seg:
                self.data[channel_id]["seg"] = None

        self.complete = self.check_complete()

    def add_raw(self, data: AcquisitionData):
        channel_id = data.channel_id
        # ensure channel id was expected
        assert channel_id in self.data, "unexpected data passed to pattern module"

        # ensure raw data was expected and not already passed
        assert "raw" in self.data[channel_id], "raw data being passed, but not expected"
        assert self.data[channel_id]["raw"] is None, (
            f"expected none, found {self.data[channel_id]['raw']}"
        )

        self.data[channel_id]["raw"] = data

    def add_seg(self, data: SegmentationData):
        channel_id = data.channel_id
        # ensure channel id was expected
        assert channel_id in self.data, "unexpected data passed to pattern module"

        # ensure raw data was expected and not already passed
        assert "seg" in self.data[channel_id], "seg data being passed, but not expected"
        assert self.data[channel_id]["seg"] is None, (
            f"expected none, found {self.data[channel_id]['seg']}"
        )

        self.data[channel_id]["seg"] = data

    def get_awaiting(self):
        awaiting = []

        for channel in self.data:
            for img in self.data[channel]:
                if self.data[channel][img] is None:
                    awaiting.append((channel, img))

        print(awaiting)

        return awaiting

    def check_complete(self):
        return len(self.get_awaiting()) == 0


class PatternContext:
    def __init__(self, data_dock: DataDock, experiment: Experiment):
        self._dock = data_dock
        self._experiment = experiment
        self._channel_map = {
            name: ch.channel_id for name, ch in experiment.channels.items()
        }

    @property
    def time(self) -> float:
        return self._dock.time_seconds

    def _get_channel_id(self, channel_name: str) -> UUID:
        if channel_name not in self._channel_map:
            raise ValueError(f"Channel '{channel_name}' not found in experiment.")
        return self._channel_map[channel_name]

    def raw(self, channel_name: str) -> np.ndarray:
        """Get raw image for a channel."""
        cid = self._get_channel_id(channel_name)
        if cid not in self._dock.data or "raw" not in self._dock.data[cid]:
            raise ValueError(
                f"Raw data for channel '{channel_name}' was not requested."
            )
        data = self._dock.data[cid]["raw"]
        return data.data if data else None

    def segmentation(self, channel_name: str) -> np.ndarray:
        """Get segmentation mask for a channel."""
        cid = self._get_channel_id(channel_name)
        if cid not in self._dock.data or "seg" not in self._dock.data[cid]:
            raise ValueError(
                f"Segmentation data for channel '{channel_name}' was not requested."
            )
        data = self._dock.data[cid]["seg"]
        return data.data if data else None


class PatternMethod:
    name = "base"

    def __init__(
        self, experiment_name=None, camera_properties: CameraProperties = None, **kwargs
    ):
        # Support legacy init where these are passed
        self.experiment_name = experiment_name
        self.camera_properties = camera_properties
        if camera_properties:
            self.pixel_size_um = camera_properties.pixel_size_um
            self.pattern_shape = (
                camera_properties.roi.height,
                camera_properties.roi.width,
            )
        else:
            self.pixel_size_um = 1.0
            self.pattern_shape = (100, 100)  # Default placeholders

        self.binning = 1
        self._requirements_list = []  # List of (channel_name, raw, seg)
        self._experiment_ref = None

    def add_requirement(self, channel_name: str, raw: bool = False, seg: bool = False):
        """Declarative way to add requirements."""
        self._requirements_list.append((channel_name, raw, seg))

    # initialize happens shortly after init
    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        # If user used add_requirement, process them
        reqs = []
        for name, needs_raw, needs_seg in self._requirements_list:
            ch = experiment.channels.get(name)
            if ch:
                reqs.append(AcquiredImageRequest(ch.channel_id, needs_raw, needs_seg))
            else:
                logger.warning(f"Pattern {self.name} requested unknown channel {name}")

        return reqs

    # configure system is run after all pattern methods are initialized
    def configure_system(
        self,
        experiment_name: str,
        camera_properties: CameraProperties,
        experiment: Experiment,
    ):
        """Called by the system to inject dependencies."""
        self.experiment_name = experiment_name
        self.camera_properties = camera_properties
        self.pixel_size_um = camera_properties.pixel_size_um
        self.pattern_shape = (camera_properties.roi.height, camera_properties.roi.width)
        self._experiment_ref = experiment

        # this binning should actually take effect
        binning = experiment.stimulation.binning
        self.update_binning(binning)

    def get_meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.pattern_shape

        y_range = np.arange(h) * self.pixel_size_um
        x_range = np.arange(w) * self.pixel_size_um

        xx, yy = np.meshgrid(x_range, y_range)

        return xx, yy

    def center_um(self) -> tuple[float, float]:
        h, w = self.pattern_shape
        return (h * self.pixel_size_um / 2.0, w * self.pixel_size_um / 2.0)

    def generate(self, data_dock: DataDock | PatternContext) -> np.ndarray:
        # If passed PatternContext, user is using new API.
        # But if user implemented old generate(self, data_dock: DataDock), we need to support that.
        # This method is called by the system.
        raise NotImplementedError

    def update_binning(self, binning: int):
        binning_rescale = binning / self.binning

        self.pixel_size_um = self.pixel_size_um * binning_rescale
        self.pattern_shape = (
            self.pattern_shape[0] // binning_rescale,
            self.pattern_shape[1] // binning_rescale,
        )

        logger.info(
            f"model {self.name} updated pixel size (um) to {self.pixel_size_um}"
        )
        logger.info(f"model {self.name} updated pattern_shape to {self.pattern_shape}")

        self.binning = binning


class PatternMethodReturnsSLM(PatternMethod):
    pass


class PatternReview(PatternMethodReturnsSLM):
    name = "pattern_review"

    def __init__(
        self,
        experiment_name,
        camera_properties,
        h5fp: str | Path | None = None,
        channel="545",
        **kwargs,
    ):
        super().__init__(experiment_name, camera_properties)

        if h5fp is None:
            raise ValueError(
                "pattern_review model requires specifying h5fp (h5 filepath)"
            )

        self.fp = Path(h5fp)
        self.channel_name = f"channel_{channel}"

        with File(str(self.fp), "r") as f:
            keys = list(f.keys())

        self.keys = natsorted(keys)

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        return []

    def generate(self, data_dock: DataDock) -> np.ndarray:
        with File(str(self.fp), "r") as f:
            while len(self.keys) > 0:
                key = self.keys.pop(0)

                if self.channel_name in f[key]:
                    return np.array(f[key]["stim_aq"]["dmd"])

        return np.array([])


if __name__ == "__main__":
    print(type(PatternMethod))
    air = AcquiredImageRequest(uuid4(), True, False)
    print(air)

# todo: handle raw and segmentation input
