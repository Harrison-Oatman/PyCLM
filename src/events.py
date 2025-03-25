from uuid import uuid4
from .experiments import DeviceProperty, ConfigGroup
from typing import Optional

class UpdatePatternEvent:

    def __init__(self, experiment, config_groups: Optional[list[ConfigGroup]] = None,
                 devices: Optional[list[ConfigGroup]] = None,):
        self.id = uuid4()

        self.experiment_name = experiment
        self.config_groups = config_groups
        self.devices = devices


class Position:

    def __init__(self, x=None, y=None, z=None, pfs=None, label=None):

        self.label = label
        self.x = x
        self.y = y
        self.z = z
        self.pfs = pfs

    def get_xy(self):
        if not ((self.x is None) or (self.y is None)):
            return [self.x, self.y]

        return None

    def get_z(self):
        return self.z

    def get_pfs(self):
        return self.get_pfs


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

    def __init__(self, experiment, position: Position,
                 scheduled_time=0, exposure_time_ms=10, needs_slm=False,
                 super_axes=None, sub_axes=None,
                 config_groups: Optional[list[ConfigGroup]] = None,
                 devices: Optional[list[ConfigGroup]] = None,
                 save_output=True,
                 do_segmentation=False, segmentation_method=None, save_segmentation=False,
                 updates_pattern=False, pattern_method=None, save_pattern=False):

        self.id = uuid4()

        # experiment (determines h5 filename)
        self.experiment_name = experiment

        # position
        self.position = position

        self.scheduled_time = scheduled_time
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

        self.updates_pattern = updates_pattern
        self.pattern_method = pattern_method
        self.save_pattern = save_pattern

    def get_rel_path(self, leading=3) -> (str, str):
        fstring = ""

        if self.super_axes is not None:
            for ax, val in enumerate(self.super_axes):

                if val is int:
                    val = str(val).zfill(leading)

                fstring += f"{val}/"

        fstring += f"{self.experiment_name}.hdf5"

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


# class PositionGrid:
#     """
#     Contains a single grid of xy(z) positions
#     """
#
#     def __init__(self, label):
#         self.label = label
#
#     def add_positions(self, positions):
