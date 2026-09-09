import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .check import CheckFailed, check_directory, find_pyclm_config
from .controller import Controller
from .core import PatternMethod, SegmentationMethod
from .core.position_mover import PositionMover
from .core.tracking import TrackingMethod
from .directories import dry_schedule_from_directory, schedule_from_directory
from .schema import PyclmConfig

logger = logging.getLogger(__name__)

# marks the handlers installed by set_logging so a later call can replace them
_PYCLM_HANDLER_FLAG = "_pyclm_run_handler"


def remove_pyclm_log_handlers():
    """Detach and close the handlers installed by a previous ``set_logging`` call."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _PYCLM_HANDLER_FLAG, False):
            root.removeHandler(handler)
            handler.close()


def set_logging(experiment_directory: Path):
    """
    Route logging for one run: WARNING and above to the console, INFO and above
    to ``<experiment_directory>/log.log``.

    Handlers from a previous call are removed first, so successive runs in the
    same interpreter each log to their own experiment directory.
    """
    remove_pyclm_log_handlers()

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(Path(experiment_directory) / "log.log")

    # Set levels for handlers
    console_handler.setLevel(logging.WARNING)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    console_format = logging.Formatter("%(levelname)s - %(message)s")
    file_format = logging.Formatter(
        "%(asctime)s - %(filename)-12s \t - %(levelname)-8s - %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    root = logging.getLogger()
    for handler in (console_handler, file_handler):
        setattr(handler, _PYCLM_HANDLER_FLAG, True)
        root.addHandler(handler)

    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def launch_gui_process(
    srcs: Sequence[tuple[str, str]],
    cwd: Path | None = None,
    experiment_dir: Path | None = None,
) -> subprocess.Popen:
    args = [
        sys.executable,
        "-m",
        "pyclm.gui.gui_controller",
        str(experiment_dir or cwd or "."),
    ]
    for fp, ch in srcs:
        args += ["--src", f"{fp}:{ch}"]
    env = os.environ.copy()
    env.setdefault("NAPARI_DISABLE_PLUGIN_AUTOLOAD", "1")
    return subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=None,
        stderr=None,
    )


def run_pyclm(
    experiment_directory,
    config_path=None,
    segmentation_methods: dict[str, type[SegmentationMethod]] | None = None,
    pattern_methods: dict[str, type[PatternMethod]] | None = None,
    position_mover: PositionMover | None = None,
    dry_image_source=None,
    tracking_methods: dict[str, type[TrackingMethod]] | None = None,
    dry: bool = False,
    gui: bool = False,
    check: bool = True,
    force: bool = False,
):
    """
    Run a pyclm experiment from a given directory and configuration file.
    :param experiment_directory: directory containing experiment files, including schedule.toml. [experiment].toml files,
                               and the position list
    :param config_path: path to pyclm_config.toml file. If None, will look for pyclm_config.toml in the experiment_directory
    :param segmentation_methods: optional dictionary of segmentation method classes to register with the SegmentationProcess
                                    key is the method name (used by [experiment].toml), value is the class
    :param pattern_methods: optional dictionary of pattern method classes to register with the PatternProcess
                                    key is the method name (used by [experiment].toml), value is the class
    :param tracking_methods: optional dictionary of tracking method classes to register with the TrackingProcess
                                    key is the method name (used by [tracking] in the experiment toml), value is the class
    :return:
    """

    experiment_directory = Path(experiment_directory)
    print(f"experiment directory: {experiment_directory}")

    # the config file: given, in the experiment directory, or in the working directory
    config_path = find_pyclm_config(experiment_directory, config_path)
    if config_path is None:
        config_path = experiment_directory / "pyclm_config.toml"
    config_path = Path(config_path)

    assert experiment_directory.exists(), (
        f"experiment directory {experiment_directory} does not exist"
    )
    assert config_path.exists(), (
        f"config file {config_path} does not exist: pyclm_config.toml must be specified or be "
        f"present in the experiment directory"
    )

    set_logging(experiment_directory)

    # the same check as `pyclm check`; errors stop the run unless forced
    if check:
        report = check_directory(
            experiment_directory,
            config_path,
            pattern_methods=pattern_methods,
            segmentation_methods=segmentation_methods,
            tracking_methods=tracking_methods,
        )
        print(report.text())
        if report.errors and not force:
            raise CheckFailed(report)

    config = PyclmConfig.from_file(config_path)
    logger.info(f"loaded config from {config_path}")

    focus_device = config.focus_device
    settle_time_s = config.settle_time_seconds
    storage_format = config.output.format
    pattern_policy = config.output.pattern_policy
    export_imagej = config.output.export_imagej
    logger.info(f"output format {storage_format}, pattern policy {pattern_policy}")

    base_path = experiment_directory

    # For dry runs without an explicit image source, discover the schedule and
    # image source together from the directory before creating the Controller.
    if dry and dry_image_source is None:
        schedule, dry_image_source = dry_schedule_from_directory(base_path)
    else:
        schedule = schedule_from_directory(base_path)

    c = Controller(
        config.config_path,
        dry,
        position_mover=position_mover,
        dry_image_source=dry_image_source,
        settle_time_s=settle_time_s,
        storage_format=storage_format,
        pattern_policy=pattern_policy,
    )

    # register any custom methods
    if segmentation_methods is not None:
        for name, method in segmentation_methods.items():
            c.register_segmentation_method(name, method)

    if pattern_methods is not None:
        for name, method in pattern_methods.items():
            c.register_pattern_method(name, method)

    if tracking_methods is not None:
        for name, method in tracking_methods.items():
            c.register_tracking_method(name, method)

    core = c.core
    core.describe()

    if focus_device:
        core.setFocusDevice(focus_device)
        logger.info(f"focus device set to '{focus_device}'")

    print("---listing available config groups---")
    for group in core.getAvailableConfigGroups():
        cg = core.getConfigGroupObject(group, False)
        print(cg.name, list(cg.items()))

    slm_shape = config.slm_shape
    at = config.affine

    c.initialize(schedule, slm_shape, at, base_path)

    all_layers = c.all_layers
    t_gcd = c.t_gcd

    all_layers_output = {
        "t": t_gcd,
        "all_layers": [f"{filepath}:{channel}" for filepath, channel in all_layers],
    }

    with open(f"{base_path}/all_layers.txt", "w") as file:
        json.dump(all_layers_output, file, indent=4)

    gui_proc = None
    if gui:
        print(all_layers)
        gui_proc = launch_gui_process(
            all_layers, cwd=base_path, experiment_dir=base_path
        )
        logger.info(f"Started GUI process (pid={gui_proc.pid})")

    c.run()

    if export_imagej:
        export_outputs(c.outbox.writer.output_paths().values())


def export_outputs(paths) -> list[Path]:
    """Write ImageJ hyperstacks next to each finished output; failures are logged, not raised."""
    from . import io as pyclm_io

    written = []
    for path in paths:
        try:
            with pyclm_io.open(path) as exp:
                written += pyclm_io.export_imagej(exp)
        except Exception as e:
            logger.error(f"ImageJ export of {path} failed: {e}", exc_info=True)
    if written:
        print(
            f"exported {len(written)} ImageJ stack(s): {', '.join(p.name for p in written)}"
        )
    return written
