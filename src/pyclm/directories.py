from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

from toml import load

from .core import ExperimentSchedule
from .core.experiments import get_config_groups, get_device_properties, ImagingConfig, ConfigGroup, \
    SegmentationConfig, PatternConfig, Experiment, PositionWithAutoFocus


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
    imaging_binning = toml_data["imaging"].get("binning", 1)
    imaging_config_groups = get_config_groups(toml_data["imaging"], "config_groups")
    imaging_device_props = get_device_properties(toml_data["imaging"], "device_properties")

    # copy base imaging config and update
    imaging_config = deepcopy(base_config)
    imaging_config.set_id()
    imaging_config.update_config_groups(imaging_config_groups)
    imaging_config.update_device_properties(imaging_device_props)
    imaging_config.exposure = imaging_exposure
    imaging_config.every_t = imaging_every_t
    imaging_config.save = imaging_save
    imaging_config.binning = imaging_binning

    channel_group = toml_data["channels"]["group"]
    presets = toml_data["channels"]["presets"]

    imaging_configs = {}
    for preset in presets:
        # copy base imaging config
        cfg = deepcopy(imaging_config)
        cfg.set_id()
        cfg.update_config_groups([ConfigGroup(channel_group, preset)])

        # get channel-specific config, if it exists
        channel_toml = toml_data["channels"].get(preset, {})

        # get channel specific exposure and t-skipping
        exposure = channel_toml.get("exposure", imaging_exposure)
        cfg.exposure = exposure

        channel_every_t = channel_toml.get("every_t", imaging_every_t)
        cfg.every_t = channel_every_t

        # channel specific binning is not allowed
        # cfg.binning = channel_toml.get("binning", imaging_binning)

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
    stimulation_config.set_id()
    stimulation_config.exposure = toml_data["stimulation"]["exposure"]
    stimulation_config.every_t = toml_data["stimulation"].get("every_t", 1)
    stimulation_config.save = toml_data["stimulation"].get("save", True)
    stimulation_config.binning = toml_data["stimulation"].get("binning", imaging_binning)

    stimulation_config.update_config_groups(get_config_groups(toml_data["stimulation"], "config_groups"))
    stimulation_config.update_device_properties(get_device_properties(toml_data["stimulation"], "device_properties"))

    # make segmentation config
    segmentation = toml_data.get("segmentation", None)

    no_seg = False

    if segmentation:
        if "method" in segmentation:
            method = segmentation.pop("method")
            segmentation_config = SegmentationConfig(method, **segmentation)

        else:
            no_seg = True

    else:
        no_seg = True

    if no_seg:
        segmentation_config = SegmentationConfig("none")


    # make pattern config
    pattern = toml_data["pattern"]
    method = pattern.pop("method")
    pattern_config = PatternConfig(method, **pattern)

    t_delay = int(toml_data.get("t_delay", 0))
    t_stop = int(toml_data.get("t_stop", 0))

    return Experiment(
        experiment_name=name,
        imaging_configs=imaging_configs,
        stimulation_config=stimulation_config,
        segmentation=segmentation_config,
        t_stop = t_stop,
        pattern=pattern_config,
        t_delay=t_delay,
    )

def read_schedule(toml_path):

    with open(toml_path, "r") as f:
        toml_data = load(f)

    toml_data = toml_data["timing"]

    out = {}

    out["t_count"] = toml_data.get("steps", 10)
    out["t_interval"] = toml_data.get("interval_seconds", 10.0)
    out["t_setup"] = toml_data.get("setup_time_seconds", 2.0)
    out["t_between"] = toml_data.get("time_between_positions", 2.0)

    return out

def positions_from_xml(fp):

    tree = ElementTree.parse(fp)
    root = tree.getroot()

    positions = []

    for node in root[0]:
        if node.attrib["runtype"] != "NDSetupMultipointListItem":
            continue

        nv = {n.tag: n.attrib["value"] for n in node if n.attrib.get("value", None) is not None}

        for tag in ["dXPosition", "dYPosition", "dZPosition", "dPFSOffset"]:
            val = nv.get(tag)
            if val is not None:
                val = float(val)

            nv[tag] = val

        if (nv["dPFSOffset"] is not None) and (nv["dPFSOffset"] < 0):
            nv["dPFSOffset"] = None

        positions.append(PositionWithAutoFocus(
            x=nv["dXPosition"],
            y=nv["dYPosition"],
            z=nv["dZPosition"],
            autofocus_offset=nv["dPFSOffset"],
            label=nv["strName"]
        ))

    return positions

def schedule_from_directory(experiment_dir: Path):
    tomls = experiment_dir.glob("*.toml")
    tomls = {f.stem: str(f) for f in tomls}

    positions_path = experiment_dir / "multipoints.xml"

    pos_list = positions_from_xml(str(positions_path))

    positions = {}
    experiments = {}

    for position in pos_list:

        name: str = position.label

        exp_stem = name.split(".")[0]

        experiment_path = tomls.get(exp_stem)

        if experiment_path is None:
            print(f"could not find {exp_stem} in {list(tomls.keys())}")
            continue

        positions[name] = position
        experiments[name] = experiment_from_toml(experiment_path, name)

    timing = read_schedule(str(experiment_dir / "schedule.toml"))

    schedule = ExperimentSchedule(experiments, positions, **timing)

    return schedule
