import numpy as np
import tifffile

from pyclm import SegmentationMethod, PatternMethod
from pyclm.core.experiments import Experiment
from pyclm.core.patterns import AcquiredImageRequest
from pyclm.core.patterns.pattern import DataDock
from pyclm.core.segmentation.cellpose_segmentation import CellposeSegmentationMethod
from pyclm import run_pyclm

from skimage.transform import resize
from skimage.measure import regionprops, label

from pathlib import Path

TEMP_FOLDER = Path(r"E:\Harrison\fly_revision_experiments\20251218d_middlevsedge\temp")

def get_temp_fp(experiment_name):

    temp_name = "_".join(experiment_name.split(".")[1:])

    return TEMP_FOLDER / f"{temp_name}_temp_mask.tif"

class EmbryoSegmentationMethod(CellposeSegmentationMethod):

    def __init__(self, experiment_name, model="embryomodel", use_gpu=True, normlow=0, normhigh=0.15, ideal_size=None, **kwargs):
        super().__init__(experiment_name, model, use_gpu, normlow, normhigh)

        if ideal_size is not None:
            self.ideal_size = ideal_size
        else:
            self.ideal_size = (100, 100)

    def segment(self, data):

        downscaled_frame = resize(data, self.ideal_size)

        tifffile.imwrite(TEMP_FOLDER / f"{self.experiment_name}_input.tif", downscaled_frame)

        print(f"data: {downscaled_frame.min()}, {downscaled_frame.max()}, {downscaled_frame.mean()}")

        out = super().segment(downscaled_frame)


        mask = out> 0
        tifffile.imwrite(TEMP_FOLDER / f"{self.experiment_name}_output.tif", mask)

        big_mask = resize(mask.astype(float), data.shape, order=1) > 0.5

        temp_fp = get_temp_fp(self.experiment_name)

        tifffile.imwrite(temp_fp, big_mask)

        return big_mask


class EmbryoOutlineSaver(PatternMethod):

    name = "outline"

    def __init__(self, experiment_name, camera_properties, embryo_channel="638", **kwargs):
        super().__init__(experiment_name, camera_properties)

        self.embryo_channel = embryo_channel
        self.seg_channel_id = None

    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:

        channel = experiment.channels.get(self.embryo_channel, None)
        assert channel is not None, f"provided channel {self.embryo_channel} is not in experiment"
        cell_seg_air = AcquiredImageRequest(channel.channel_id, False, True)

        self.seg_channel_id = channel.channel_id

        return [cell_seg_air]

    def generate(self, data_dock: DataDock) -> np.ndarray:

        # print(data_dock.data)
        #
        # seg = data_dock.data[self.seg_channel_id]["seg"].data
        #
        # tifffile.imwrite(get_temp_fp(self.experiment), seg)

        xx, _ = self.get_meshgrid()

        return xx*0

class PatternAlongAxis(PatternMethod):

    def __init__(self, experiment_name, camera_properties, **kwargs):

        super().__init__(experiment_name, camera_properties)

        self.mask = None
        self.centroid = None
        self.long_axis = None
        self.axis_length = None

    def check_for_mask(self) -> bool:

        temp_fp = get_temp_fp(self.experiment)

        if not temp_fp.exists():
            return False

        mask = tifffile.imread(temp_fp)
        self.mask = mask

        labeled_mask = label(mask)

        props = regionprops(labeled_mask)

        biggest_prop_area = 0

        if len(props) == 0:
            return False

        for prop in props:
            if prop.area > biggest_prop_area:
                self.mask = labeled_mask == prop.label
                self.centroid = prop.centroid
                self.long_axis = (np.sin(prop.orientation), np.cos(prop.orientation))
                self.axis_length = prop.axis_major_length


        return True

    def generate(self, data):
        raise NotImplementedError

class InnerPatternMethod(PatternAlongAxis):

    def __init__(self, experiment_name, camera_properties, fraction_length=0.1, **kwargs):
        super().__init__(experiment_name, camera_properties)

        self.fraction_length = fraction_length

    def generate(self, data):

        if not self.check_for_mask():
            return 0 * self.get_meshgrid()[0]

        y_arange = np.arange(self.pattern_shape[0])
        x_arange = np.arange(self.pattern_shape[1])

        yy, xx = np.meshgrid(y_arange, x_arange)

        mag = (yy - self.centroid[1]) * self.long_axis[0] + (xx - self.centroid[0]) * self.long_axis[1]

        mag = np.abs(mag) / (self.axis_length / 2)

        return (mag < self.fraction_length) * self.mask


class OuterPatternMethod(PatternAlongAxis):

    def __init__(self, experiment_name, camera_properties, fraction_length=0.1, **kwargs):
        super().__init__(experiment_name, camera_properties)

        self.fraction_length = fraction_length

    def generate(self, data):
        if not self.check_for_mask():
            return 0 * self.get_meshgrid()[0]

        y_arange = np.arange(self.pattern_shape[0])
        x_arange = np.arange(self.pattern_shape[1])

        yy, xx = np.meshgrid(y_arange, x_arange)

        mag = (yy - self.centroid[1]) * self.long_axis[0] + (xx - self.centroid[0]) * self.long_axis[1]

        mag = np.abs(mag) / (self.axis_length / 2)

        return (mag > (1 - self.fraction_length)) * self.mask

segmentation_methods = {
    "embryo": EmbryoSegmentationMethod
}

pattern_methods = {
    "outline": EmbryoOutlineSaver,
    "inner": InnerPatternMethod,
    "outer": OuterPatternMethod,
}

if __name__ == "__main__":

    experiment_directory = Path(r"E:\Harrison\fly_revision_experiments\20251218d_middlevsedge")

    run_pyclm(experiment_directory, segmentation_methods=segmentation_methods, pattern_methods=pattern_methods)



