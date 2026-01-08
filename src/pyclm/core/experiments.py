"""
Defines the Experiment class and related data structures for managing imaging experiments.
"""

from collections import namedtuple
from typing import Optional
from uuid import uuid4

ConfigGroup = namedtuple("ConfigGroup", ["group", "config"])
DeviceProperty = namedtuple("DeviceProperty", ["device", "property", "value", "type"])


def make_configgroup_dict(config_groups: list[ConfigGroup] | None):
    if config_groups is None:
        return {}

    return {
        config: cfg
        for (config, _), cfg in zip(config_groups, config_groups, strict=False)
    }


def make_deviceproperty_dict(device_properties: list[DeviceProperty] | None):
    if device_properties is None:
        return {}

    return {
        f"{dev}-{prop}": devprop
        for (dev, prop, _, _), devprop in zip(
            device_properties, device_properties, strict=False
        )
    }


class ImagingConfig:
    """
    Contains device properties and config groups which can be updated during experiment initialization or runtime
    """

    def __init__(
        self,
        experiment_name: str,
        exposure_ms: float = 10,
        every_t: int = 1,
        binning: int = 1,
        config_groups: list[ConfigGroup] | None = None,
        device_properties: list[DeviceProperty] | None = None,
        save=True,
    ):
        self.channel_id = uuid4()
        self.experiment_name = experiment_name
        self.exposure = exposure_ms
        self.every_t = every_t
        self.save = save
        self.binning = binning

        self._config_groups = make_configgroup_dict(config_groups)
        self._device_properties = make_deviceproperty_dict(device_properties)

    def update_config_groups(self, gps: list[ConfigGroup]):
        gps = make_configgroup_dict(gps)
        self._config_groups.update(gps)

    def update_device_properties(self, devp: list[DeviceProperty]):
        devp = make_deviceproperty_dict(devp)
        self._device_properties.update(devp)

    def get_config_groups(self) -> list[ConfigGroup]:
        return list(self._config_groups.values())

    def get_device_properties(self) -> list[DeviceProperty]:
        return list(self._device_properties.values())

    def set_id(self):
        self.channel_id = uuid4()

    def __repr__(self):
        return (
            f"ImagingConfig({self.channel_id}: exposure={self.exposure}ms, image_every={self.every_t}, "
            f"configs={self._config_groups}, device_props={self._device_properties})"
        )


class MethodBasedConfig:
    def __init__(self, method_name: str, save_output: bool = True, every_t=1, **kwargs):
        self.method_name = method_name
        self.save = save_output
        self.every_t = every_t

        print(f"method kwargs: {kwargs}")

        self.kwargs = kwargs

    def __repr__(self):
        return f"Method({self.method_name}: {', '.join([f'{k}={v}' for k, v in self.kwargs.items()])})"


SegmentationConfig = MethodBasedConfig
PatternConfig = MethodBasedConfig


class Experiment:
    def __init__(
        self,
        experiment_name,
        imaging_configs: dict[str, ImagingConfig],
        stimulation_config: ImagingConfig,
        segmentation: SegmentationConfig,
        pattern: PatternConfig,
        t_delay: int = 0,
        t_stop: int = 0,
    ):
        self.key = uuid4()
        self.experiment_name = experiment_name
        self.channels = imaging_configs
        self.stimulation = stimulation_config
        self.segmentation = segmentation
        self.pattern = pattern

        self.t_delay = t_delay
        self.t_stop = t_stop

    def __repr__(self):
        return (
            f"Experiment('{self.experiment_name}': Channels={self.channels}, Stimulation={self.stimulation}, "
            f"Segmentation method={self.segmentation}, Pattern method={self.pattern})"
        )


class PositionBase:
    pass


class PositionWithAutoFocus(PositionBase):
    def __init__(self, x=None, y=None, z=None, autofocus_offset=None, label=None):
        self.label = label
        self.x = x
        self.y = y
        self.z = z
        self.autofocus_offset = autofocus_offset

    def get_xy(self):
        if not ((self.x is None) or (self.y is None)):
            return [self.x, self.y]

        return None

    def get_z(self):
        return self.z

    def get_autofocus_offset(self):
        return self.autofocus_offset

    def as_dict(self):
        return {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "autofocus_offset": self.autofocus_offset,
        }


class TimeCourse:
    def __init__(self, count, interval, setup, between):
        self.count = count
        self.interval = interval
        self.setup = setup
        self.between = between


class ExperimentSchedule:
    def __init__(
        self,
        experiments: dict[str:Experiment],
        positions: dict[str, PositionBase],
        t_count: int = 1,
        t_interval: float = 30.0,
        t_setup=2.0,
        t_between=1.0,
        timecourse: TimeCourse | None = None,
    ):
        self.experiment_names = [exp for exp in experiments]
        self.experiments = experiments
        self.positions = positions

        assert set(experiments.keys()) == set(positions.keys())

        if timecourse:
            self.times = timecourse

        else:
            self.times = TimeCourse(t_count, t_interval, t_setup, t_between)


def check_type(obj):
    if isinstance(obj, float):
        return "float"
    elif isinstance(obj, str):
        return "str"
    elif isinstance(obj, int):
        return "int"
    elif isinstance(obj, bool):
        return "bool"
    else:
        return "unknown"


def get_config_groups(toml_dict, key):
    """
    Returns a list of config groups from the toml_dict
    :param toml_dict: dict
    :param key: str
    :return: list[ConfigGroup]
    """
    if key not in toml_dict:
        return []

    config_groups = []
    for group, config in toml_dict[key].items():
        config_groups.append(ConfigGroup(group, config))

    return config_groups


def get_device_properties(toml_dict, key):
    """
    Returns a list of device properties from the toml_dict
    :param toml_dict: dict
    :param key: str
    :return: list[DeviceProperty]
    """
    if key not in toml_dict:
        return []

    device_properties = []
    for k, v in toml_dict[key].items():
        try:
            dev, prop = k.split("-")[0], k.split("-")[1]
        except IndexError as e:
            raise IndexError(
                f"error in device property {k}; should be formatted as device-property"
            ) from e

        t = check_type(v)
        device_properties.append(DeviceProperty(dev, prop, v, t))

    return device_properties


# todo: generate positions from micromanager output
# todo: make grid-based acquisition
