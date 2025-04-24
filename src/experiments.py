from collections import namedtuple
from typing import Optional
from toml import load
from copy import deepcopy
from uuid import uuid4

ConfigGroup = namedtuple("ConfigGroup", ["group", "config"])
DeviceProperty = namedtuple("DeviceProperty", ["device", "property", "value", "type"])


def make_configgroup_dict(config_groups: Optional[list[ConfigGroup]]):
    if config_groups is None:
        return {}

    return {config: cfg for (config, _), cfg in zip(config_groups, config_groups)}


def make_deviceproperty_dict(device_properties: Optional[list[DeviceProperty]]):
    if device_properties is None:
        return {}

    return {f"{dev}-{prop}": devprop for (dev, prop, _, _), devprop in zip(device_properties, device_properties)}


class ImagingConfig:
    """
    Contains device properties and config groups which can be updated during experiment initialization or runtime
    """

    def __init__(self, experiment_name: str, exposure_ms: float = 10, every_t: int = 1,
                 config_groups: Optional[list[ConfigGroup]] = None,
                 device_properties: Optional[list[DeviceProperty]] = None,
                 save=True):
        self.channel_id = uuid4()
        self.experiment_name = experiment_name
        self.exposure = exposure_ms
        self.every_t = every_t
        self.save = save

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

    def __repr__(self):
        return (f"ImagingConfig(exposure={self.exposure}ms, image_every={self.every_t}, "
                f"configs={self._config_groups}, device_props={self._device_properties})")


class MethodBasedConfig:

    def __init__(self, method_name: str, save_output: bool = True, every_t = 1, **kwargs):
        self.method_name = method_name
        self.save = save_output
        self.every_t = every_t

        self.kwargs = kwargs

    def __repr__(self):
        return f"Method({self.method_name}: {', '.join([f'{k}={v}' for k, v in self.kwargs.items()])})"


SegmentationConfig = MethodBasedConfig
PatternConfig = MethodBasedConfig


class Experiment:

    def __init__(self, experiment_name, imaging_configs: dict[str, ImagingConfig], stimulation_config: ImagingConfig,
                 segmentation: SegmentationConfig, pattern: PatternConfig):
        self.key = uuid4()
        self.experiment_name = experiment_name
        self.channels = imaging_configs
        self.stimulation = stimulation_config
        self.segmentation = segmentation
        self.pattern = pattern

    def __repr__(self):
        return (f"Experiment('{self.experiment_name}': Channels={self.channels}, Stimulation={self.stimulation}, "
                f"Segmentation method={self.segmentation}, Pattern method={self.pattern})")


class PositionBase:
    pass


class Position(PositionBase):

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
        return self.pfs

    def as_dict(self):
        return {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "pfs": self.pfs
        }


class TimeCourse:

    def __init__(self, count, interval, setup, between):
        self.count = count
        self.interval = interval
        self.setup = setup
        self.between = between


class ExperimentSchedule:

    def __init__(self, experiments: dict[str: Experiment], positions: dict[str, PositionBase],
                 t_count: int = 1, t_interval: float = 30.0, t_setup=2., t_between=1.,
                 timecourse: Optional[TimeCourse] = None):

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
        except IndexError:
            raise IndexError(f"error in device property {k}; should be formatted as device-property")

        t = check_type(v)
        device_properties.append(DeviceProperty(dev, prop, v, t))

    return device_properties


def experiment_from_toml(toml_path, name="SampleExperiment"):
    with open(toml_path, "r") as f:
        toml_data = load(f)

    base_config_groups = get_config_groups(toml_data, "config_groups")
    base_device_props = get_device_properties(toml_data, "device_properties")

    base_config = ImagingConfig(
        experiment_name=name,
        config_groups=base_config_groups,
        device_properties=base_device_props,
    )

    imaging_exposure = toml_data["imaging"].get("exposure", 10)
    imaging_every_t = toml_data["imaging"].get("every_t", 1)
    imaging_save = toml_data["imaging"].get("save", True)
    imaging_config_groups = get_config_groups(toml_data["imaging"], "config_groups")
    imaging_device_props = get_device_properties(toml_data["imaging"], "device_properties")

    # copy base imaging config and update
    imaging_config = deepcopy(base_config)
    imaging_config.update_config_groups(imaging_config_groups)
    imaging_config.update_device_properties(imaging_device_props)
    imaging_config.exposure = imaging_exposure
    imaging_config.every_t = imaging_every_t
    imaging_config.save = imaging_save

    channel_group = toml_data["channels"]["group"]
    presets = toml_data["channels"]["presets"]

    imaging_configs = {}
    for preset in presets:
        # copy base imaging config
        cfg = deepcopy(imaging_config)
        cfg.update_config_groups([ConfigGroup(channel_group, preset)])

        # get channel-specific config, if it exists
        channel_toml = toml_data["channels"].get(preset, {})

        # get channel specific exposure and t-skipping
        exposure = channel_toml.get("exposure", imaging_exposure)
        cfg.exposure = exposure

        channel_every_t = channel_toml.get("every_t", imaging_every_t)
        cfg.every_t = channel_every_t

        # get channel-specific config groups and device properties
        device_properties = channel_toml.get("device_properties", {})
        device_properties = get_device_properties(device_properties, "device_properties")
        cfg.update_device_properties(device_properties)

        config_groups = channel_toml.get("config_groups", {})
        config_groups = get_config_groups(config_groups, "config_groups")
        cfg.update_config_groups(config_groups)

        # get channel-specific segmentation and pattern configs
        imaging_configs[preset] = cfg

    # make stimulation config - inherits base config
    stimulation_config = deepcopy(base_config)
    stimulation_config.exposure = toml_data["stimulation"]["exposure"]
    stimulation_config.every_t = toml_data["stimulation"].get("every_t", 1)
    stimulation_config.save = toml_data["stimulation"].get("save", True)

    stimulation_config.update_config_groups(get_config_groups(toml_data["stimulation"], "config_groups"))
    stimulation_config.update_device_properties(get_device_properties(toml_data["stimulation"], "device_properties"))

    # make segmentation config
    segmentation = toml_data["segmentation"]
    method = segmentation.pop("method")
    segmentation_config = SegmentationConfig(method, **segmentation)

    # make pattern config
    pattern = toml_data["pattern"]
    method = pattern.pop("method")
    pattern_config = PatternConfig(method, **pattern)

    return Experiment(
        experiment_name=name,
        imaging_configs=imaging_configs,
        stimulation_config=stimulation_config,
        segmentation=segmentation_config,
        pattern=pattern_config,
    )

# todo: generate ExperimentSchedule from toml
# todo: write sample experimentschedule.toml
# todo: generate positions from elements output
# todo: generate positions from micromanager output
# todo: make grid-based acquisition


if __name__ == "__main__":
    print(experiment_from_toml(r"/experiments/SampleExperiment.toml"))
