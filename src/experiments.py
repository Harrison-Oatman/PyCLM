from collections import namedtuple
from typing import Optional

ConfigGroup = namedtuple("ConfigGroup", ["group", "config"])
DeviceProperty = namedtuple("DeviceProperty", ["device", "property", "value", "type"])


def configgroup_dict(config_groups: Optional[list[ConfigGroup]]):
    if config_groups is None:
        return {}

    return {config: cfg for (config, _), cfg in zip(config_groups, config_groups)}


def deviceproperty_dict(device_properties: Optional[list[DeviceProperty]]):
    if device_properties is None:
        return {}

    return {f"{dev}-{prop}": devprop for (dev, prop, _, _), devprop in zip(device_properties, device_properties)}


class ImagingConfig:
    """
    Contains device properties and config groups which can be updated during experiment initialization or runtime
    """

    def __init__(self, name: str, exposure_ms: float = 10,
                 config_groups: Optional[list[ConfigGroup]] = None,
                 device_properties: Optional[list[DeviceProperty]] = None):

        self.name = name
        self.exposure = exposure_ms

        self.config_groups = configgroup_dict(config_groups)
        self.device_properties = deviceproperty_dict(device_properties)

    def update_config_groups(self, gps: list[ConfigGroup]):

        gps = configgroup_dict(gps)
        self.config_groups.update(gps)

    def update_device_properties(self, devp: list[DeviceProperty]):

        devp = deviceproperty_dict(devp)
        self.device_properties.update(devp)


class MethodBasedConfig:

    def __init__(self, method_name: str, save_output: bool = True, **kwargs):

        self.method_name = method_name
        self.save = save_output

        self.kwargs = kwargs


SegmentationConfig = MethodBasedConfig
PatternConfig = MethodBasedConfig


class Experiment:

    def __init__(self, experiment_name, imaging_configs: list[ImagingConfig], stimulation_config: ImagingConfig,
                 segmentation: SegmentationConfig, pattern: PatternConfig):

        self.experiment_name = experiment_name
        self.channels = imaging_configs
        self.stimulation = stimulation_config
        self.segmentation = segmentation
        self.pattern = pattern

