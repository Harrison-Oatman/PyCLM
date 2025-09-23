from typing import Optional
import logging
from toml import load
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

from .controller import Controller
from src.pyclm.core.segmentation import SegmentationMethod
from src.pyclm.core.patterns import PatternMethod
from .directories import schedule_from_directory


def run_pyclm(experiment_directory, config_path=None,
              segmentation_methods: Optional[dict[str, type[SegmentationMethod]]]= None,
              pattern_methods: Optional[dict[str, type[PatternMethod]]]= None):
    """
    Run a pyclm experiment from a given directory and configuration file.
    :param experiment_directory: directory containing experiment files, including schedule.toml. [experiment].toml files,
                               and the position list
    :param config_path: path to pyclm_config.toml file. If None, will look for pyclm_config.toml in the experiment_directory
    :param segmentation_methods: optional dictionary of segmentation method classes to register with the SegmentationProcess
                                    key is the method name (used by [experiment].toml), value is the class
    :param pattern_methods: optional dictionary of pattern method classes to register with the PatternProcess
                                    key is the method name (used by [experiment].toml), value is the class
    :return:
    """

    experiment_directory = Path(experiment_directory)

    # search for config file if not provided
    if config_path is None:
        # look in the experiment directory for pyclm_config.toml
        config_path = experiment_directory / "pyclm_config.toml"

        # look in the current working directory for pyclm_config.toml
        if not config_path.exists():
            config_path = Path("pyclm_config.toml")

    config_path = Path(config_path)

    assert experiment_directory.exists(), f"experiment directory {experiment_directory} does not exist"
    assert config_path.exists(), (f"config file {config_path} does not exist: pyclm_config.toml must be specified or be "
                                  f"present in the experiment directory")

    set_logging(experiment_directory)

    config = load(config_path)
    logger.info(f"loaded config from {config_path}")

    base_path = experiment_directory

    c = Controller(config["config_path"])

    # register any custom methods
    if segmentation_methods is not None:
        for name, method in segmentation_methods.items():
            c.register_segmentation_method(name, method)

    if pattern_methods is not None:
        for name, method in pattern_methods.items():
            c.register_pattern_method(name, method)

    core = c.core
    core.describe()

    core.setFocusDevice("ZDrive")

    print("---listing available config groups---")
    for group in core.getAvailableConfigGroups():
        cg = core.getConfigGroupObject(group, False)
        print(cg.name, list(cg.items()))

    schedule = schedule_from_directory(base_path)

    slm_shape = config["slm_shape_h"], config["slm_shape_w"]
    at = np.array(config["affine_transform"], dtype=np.float32)

    c.initialize(schedule, slm_shape, at, base_path)

    c.run()