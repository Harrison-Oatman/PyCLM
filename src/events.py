from uuid import uuid4, UUID
from .experiments import ConfigGroup, Position, DeviceProperty
from typing import Optional
from h5py import Dataset
import datetime



class UpdatePatternEvent:

    def __init__(self, experiment, config_groups: Optional[list[ConfigGroup]] = None,
                 devices: Optional[list[DeviceProperty]] = None, ):
        self.id = uuid4()

        self.experiment_name = experiment
        self.config_groups = config_groups
        self.devices = devices


class UpdateStagePositionEvent:
    """
    Moves the stage
    """

    def __init__(self, position):
        self.id = uuid4()
        self.position = position


class GeneratePatternEvent:

    def __init__(self, experiment, model, uses_acquisition=False, acquisition_event_id=None,
                 uses_segmentation=False, segmentation_event_id=None,
                 save_output=True, **kwargs):
        self.id = uuid4()

        self.experiment_name = experiment
        self.model = model
        self.uses_acquisition = uses_acquisition
        self.acquisition_event_id = acquisition_event_id
        self.save_output = save_output
        self.kwargs = kwargs


class AcquisitionEvent:

    def __init__(self, experiment, position: Position, channel_id: UUID,
                 scheduled_time=0, scheduled_time_since_start=0, exposure_time_ms=10, needs_slm=False,
                 super_axes=None, sub_axes=None,
                 config_groups: Optional[list[ConfigGroup]] = None,
                 devices: Optional[list[DeviceProperty]] = None,
                 save_output=True,
                 do_segmentation=False, segmentation_method=None, save_segmentation=False,
                 raw_goes_to_pattern=False, pattern_method=None, save_pattern=False,
                 segmentation_goes_to_pattern=False,
                 ):

        self.id = uuid4()

        # experiment (determines h5 filename)
        self.experiment_name = experiment

        # position
        self.position = position

        self.scheduled_time = scheduled_time
        self.time_since_start = scheduled_time_since_start
        self.complete = False
        self.completed_time = None

        # acquisition details
        self.exposure_time_ms = exposure_time_ms
        self.needs_slm = needs_slm

        # axis-name, axis-value pairs
        # super-axes (determines folder structure containing experiment)
        self.super_axes = super_axes

        # sub-axes (determines folder within hdf5_file)
        self.sub_axes = sub_axes

        # config group config-preset pairs
        self.config_groups = config_groups

        # device-name, parameter, value, type
        self.devices = devices

        # what to do with the output
        self.save_output = save_output

        self.segment = do_segmentation
        self.seg_method = segmentation_method
        self.save_seg = save_segmentation

        self.raw_goes_to_pattern = raw_goes_to_pattern
        self.seg_goes_to_pattern = segmentation_goes_to_pattern
        self.channel_id = channel_id

        self.pattern_method = pattern_method
        self.save_pattern = save_pattern

    def get_rel_path(self, leading=3) -> (str, str):
        fstring = ""

        if self.super_axes is not None:
            for ax, val in enumerate(self.super_axes):

                if val is int:
                    val = str(val).zfill(leading)

                fstring += f"{val}/"

        fstring += f"{self.experiment_name}"

        dset = ""

        if self.sub_axes is not None:
            for ax, val in enumerate(self.sub_axes):

                if val is int:
                    val = str(val).zfill(leading)

                dset += f"{val}/"

            dset.rstrip("/")

        else:
            dset = "UNNAMED_DATA"

        return fstring, dset

    def write_attrs(self, dset: Dataset):
        dset.attrs["id"] = str(self.id)
        dset.attrs["position"] = [(k, str(v)) for k, v in self.position.as_dict().items()]

        dset.attrs["experiment_name"] = self.experiment_name

        value = datetime.datetime.fromtimestamp(self.scheduled_time)
        dset.attrs["time_scheduled"] = value.strftime('%Y-%m-%d %H:%M:%S')
        dset.attrs["time_since_start"] = str(datetime.timedelta(seconds=self.time_since_start))
        dset.attrs["time_completed"] = datetime.datetime.fromtimestamp(self.completed_time).strftime('%Y-%m-%d %H:%M:%S')
        dset.attrs["complete"] = self.complete

        dset.attrs["exposure_time_ms"] = self.exposure_time_ms
        dset.attrs["needs_slm"] = self.needs_slm

        if self.super_axes is not None:
            dset.attrs["super_axes"] = [str(a) for a in self.super_axes]

        if self.sub_axes is not None:
             dset.attrs["sub_axes"] = [str(a) for a in self.sub_axes]

        if self.config_groups is not None:
            for cg in self.config_groups:
                dset.attrs[f"config_groups: {cg.group}"] = str(cg.config)

        if self.devices is not None:
            for dp in self.devices:
                dset.attrs[f"devices: {dp.device}-{dp.property}"] = str(dp.value)

        dset.attrs["save_output"] = self.save_output
        dset.attrs["segment"] = self.segment
        dset.attrs["seg_method"] = self.seg_method
        dset.attrs["save_seg"] = self.save_seg

        dset.attrs["raw_goes_to_pattern"] = self.raw_goes_to_pattern
        dset.attrs["seg_goes_to_pattern"] = self.seg_goes_to_pattern
        dset.attrs["channel_id"] = str(self.channel_id)

        dset.attrs["pattern_method"] = self.pattern_method
        dset.attrs["save_pattern"] = self.save_pattern

# class PositionGrid:
#     """
#     Contains a single grid of xy(z) positions
#     """
#
#     def __init__(self, label):
#         self.label = label
#
#     def add_positions(self, positions):
