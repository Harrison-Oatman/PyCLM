from ..experiments import Experiment
from ..datatypes import AcquisitionData, SegmentationData
import logging
from typing import NamedTuple, Union
from uuid import uuid4, UUID
from collections import defaultdict
from pathlib import Path
from h5py import File
import numpy as np
from natsort import natsorted

logger = logging.getLogger(__name__)

ROI = NamedTuple("ROI", [("x_offset", int), ("y_offset", int), ("width", int), ("height", int)])
CameraProperties = NamedTuple("CameraProperties", [("roi", ROI), ("pixel_size_um", float)])
AcquiredImageRequest = NamedTuple("AcquiredImageRequest", [("id", UUID), ("needs_raw", bool), ("needs_seg", bool)])


class DataDock:

    def __init__(self, t, requirements: list[AcquiredImageRequest]):

        self.time_seconds = t
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
        assert self.data[channel_id]["raw"] is None, f"expected none, found {self.data[channel_id]['raw']}"

        self.data[channel_id]["raw"] = data

    def add_seg(self, data: SegmentationData):

        channel_id = data.channel_id
        # ensure channel id was expected
        assert channel_id in self.data, "unexpected data passed to pattern module"

        # ensure raw data was expected and not already passed
        assert "seg" in self.data[channel_id], "seg data being passed, but not expected"
        assert self.data[channel_id]["seg"] is None, f"expected none, found {self.data[channel_id]['seg']}"

        self.data[channel_id]["seg"] = data

    def get_awaiting(self):

        awaiting = []

        for channel in self.data:
            for img in self.data[channel]:
                if self.data[channel][img] is None:
                    awaiting.append((channel, img))

        return awaiting

    def check_complete(self):

        return len(self.get_awaiting()) == 0


class PatternModel:

    name = "base"

    def __init__(self, experiment_name, camera_properties: CameraProperties, **kwargs):
        self.experiment = experiment_name
        self.camera_properties = camera_properties
        self.pixel_size_um = camera_properties.pixel_size_um
        self.pattern_shape = (camera_properties.roi.height, camera_properties.roi.width)
        self.binning = 1

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:

        binning = experiment.stimulation.binning

        self.update_binning(binning)

        return []

    def get_meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.pattern_shape

        y_range = np.arange(h) * self.pixel_size_um
        x_range = np.arange(w) * self.pixel_size_um

        xx, yy = np.meshgrid(x_range, y_range)

        return xx, yy

    def generate(self, data_dock: DataDock) -> np.ndarray:
        raise NotImplementedError

    def update_binning(self, binning: int):

        binning_rescale = binning / self.binning

        self.pixel_size_um = self.pixel_size_um * binning_rescale
        self.pattern_shape = (self.pattern_shape[0] // binning_rescale, self.pattern_shape[1] // binning_rescale)

        logger.info(f"model {self.name} updated pixel size (um) to {self.pixel_size_um}")
        logger.info(f"model {self.name} updated pattern_shape to {self.pattern_shape}")

        self.binning = binning


class PatternModelReturnsSLM(PatternModel):

    pass


class PatternReview(PatternModelReturnsSLM):

    name = "pattern_review"

    def __init__(self, experiment_name, camera_properties, h5fp: Union[str, Path] = None, channel="545", **kwargs):
        super().__init__(experiment_name, camera_properties)

        if h5fp is None:
            raise ValueError("pattern_review model requires specifying h5fp (h5 filepath)")

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
    print(type(PatternModel))
    air = AcquiredImageRequest(uuid4(), True, False)
    print(air)

# todo: handle raw and segmentation input
