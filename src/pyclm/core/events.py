import datetime
import json
from uuid import UUID, uuid4

from h5py import Dataset

from .experiments import (
    ConfigGroup,
    DeviceProperty,
    MicroscopePosition,
    PositionBase,
    PositionWithAutoFocus,
)

STIM_GROUP = "stim_aq"


def storage_group(channel: str, is_stim: bool) -> str:
    """HDF5 group for a channel: ``stim_aq`` for stimulation, ``channel_<name>`` otherwise."""
    return STIM_GROUP if is_stim else f"channel_{channel}"


def rel_path(index: dict, is_stim: bool) -> str:
    """Dataset path within the HDF5 file for a frame index, e.g. ``00012/channel_545/``."""
    return f"{index['t']:05d}/{storage_group(index['c'], is_stim)}/"


class UpdatePatternEvent:
    def __init__(
        self,
        experiment,
        config_groups: list[ConfigGroup] | None = None,
        devices: list[DeviceProperty] | None = None,
    ):
        self.id = uuid4()

        self.experiment_name = experiment
        self.config_groups = config_groups
        self.devices = devices


class UpdateStagePositionEvent:
    """
    Moves the stage
    """

    def __init__(self, position: PositionBase, experiment_name: str):
        self.id = uuid4()
        self.position = position
        self.experiment_name = experiment_name


class UpdatePositionWithAutoFocusEvent(UpdateStagePositionEvent):
    def __init__(self, position: PositionWithAutoFocus, experiment_name):
        super().__init__(position, experiment_name)


class AcquisitionEvent:
    """
    One image acquisition: identity from the plan (``index``), where and how
    to take it, and whether the frame is saved. Who consumes the frame is
    not the event's business; the Router decides that from its identity.
    """

    def __init__(
        self,
        experiment,
        position: MicroscopePosition,
        channel_id: UUID,
        index: dict | None = None,
        scheduled_time=0,
        scheduled_time_since_start=0,
        exposure_time_ms=10,
        needs_slm=False,
        config_groups: list[ConfigGroup] | None = None,
        devices: list[DeviceProperty] | None = None,
        save_output=True,
        binning: int = 1,
    ):
        self.id = uuid4()

        # experiment (determines the output file / store)
        self.experiment_name = experiment

        # position
        self.position = position

        self.scheduled_time = scheduled_time
        self.time_since_start = scheduled_time_since_start
        self.complete = False
        self.completed_time = None

        # frame identity from the plan: {"t": timepoint, "p": experiment, "c": channel}
        self.index: dict = dict(index) if index else {}

        # acquisition details
        self.exposure_time_ms = exposure_time_ms
        self.needs_slm = needs_slm
        self.binning = binning

        # config group config-preset pairs
        self.config_groups = config_groups

        # device-name, parameter, value, type
        self.devices = devices

        # whether the writer records the frame
        self.save_output = save_output

        self.channel_id = channel_id

        # settings changed at runtime for this channel and in force for this
        # frame, {frames-table column: value} (see core/settings.py)
        self.overrides: dict = {}

        self.pixel_width_um = None

    @property
    def t_index(self) -> int:
        return int(self.index.get("t", 0))

    def get_rel_path(self) -> str:
        """
        Returns dset path within the hdf5 structure
        """
        if "t" not in self.index or "c" not in self.index:
            return "UNNAMED_DATA/"
        return rel_path(self.index, self.needs_slm)

    def _fmt_time(self, timestamp) -> str:
        if timestamp is None:
            return ""
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def as_attrs(self) -> dict:
        """
        Flat, HDF5-attribute-compatible description of this event.

        Used both to annotate datasets (``write_attrs``) and for ``repr``.
        """
        attrs = {}

        attrs["id"] = str(self.id)
        attrs["position"] = [(k, str(v)) for k, v in self.position.as_dict().items()]

        attrs["experiment_name"] = self.experiment_name

        attrs["time_scheduled"] = self._fmt_time(self.scheduled_time)
        attrs["time_since_start"] = str(
            datetime.timedelta(seconds=self.time_since_start)
        )
        attrs["time_completed"] = self._fmt_time(self.completed_time)
        attrs["complete"] = self.complete

        attrs["exposure_time_ms"] = self.exposure_time_ms
        attrs["needs_slm"] = self.needs_slm
        attrs["binning"] = self.binning

        attrs["index"] = json.dumps(self.index)

        if self.config_groups is not None:
            for cg in self.config_groups:
                attrs[f"config_groups: {cg.group}"] = str(cg.config)

        if self.devices is not None:
            for dp in self.devices:
                attrs[f"devices: {dp.device}-{dp.property}"] = str(dp.value)

        attrs["save_output"] = self.save_output
        attrs["channel_id"] = str(self.channel_id)

        attrs["pixel_width_um"] = str(self.pixel_width_um)

        return attrs

    def write_attrs(self, dset: Dataset):
        for key, value in self.as_attrs().items():
            dset.attrs[key] = value

    def __repr__(self):
        return repr(self.as_attrs())
