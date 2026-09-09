"""
The ``pyclm`` command.

::

    pyclm run <dir> [--config FILE] [--dry] [--gui] [--no-check | --force]
    pyclm check <dir> [--config FILE] [--mm-config FILE] [--dry]
    pyclm preview <dir> <experiment> (--image FILE | --snap) [--config FILE] [--out DIR] [--t N]
    pyclm export <dir> [channels ...] [--config FILE]
    pyclm gui <dir>
    pyclm control <dir> [--dry] [--config FILE]
    pyclm new <dir> [--template open-loop|closed-loop] [--name NAME]

``pyclm <dir> [--dry] [--gui]`` (the form before Stage 5) still means ``run``.
Custom methods need the Python entry points (``run_pyclm``,
``check_directory``, ``preview``), which take the same dictionaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMMANDS = ("run", "check", "preview", "export", "gui", "new", "control")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyclm", description="Closed-loop optogenetic microscopy experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="check an experiment directory, then run it")
    p.add_argument("directory", help="directory containing the experiment files")
    p.add_argument("--config", default=None, help="path to pyclm_config.toml")
    p.add_argument("--dry", action="store_true", help="run on the virtual microscope")
    p.add_argument("--gui", action="store_true", help="open the live viewer")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no-check", action="store_true", help="skip the check")
    g.add_argument(
        "--force", action="store_true", help="run even if the check finds errors"
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="report problems in an experiment directory")
    p.add_argument("directory")
    p.add_argument("--config", default=None, help="path to pyclm_config.toml")
    p.add_argument(
        "--mm-config", default=None, help="MicroManager .cfg to check presets against"
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="also run two timepoints on the virtual microscope",
    )
    p.set_defaults(func=cmd_check)

    p = sub.add_parser(
        "preview",
        help="run one experiment's segmentation and pattern method on one image",
    )
    p.add_argument("directory")
    p.add_argument(
        "experiment", help="a position label (bar10.00) or a TOML stem (bar10)"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", default=None, help="a TIF to use for every channel")
    src.add_argument("--snap", action="store_true", help="snap from the microscope")
    p.add_argument("--config", default=None, help="path to pyclm_config.toml")
    p.add_argument(
        "--out", default=None, help="output directory (default: <dir>/preview/<label>)"
    )
    p.add_argument("--t", type=int, default=0, help="the timepoint to pretend it is")
    p.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="pixel size of --image (default 1.0)",
    )
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("export", help="write ImageJ hyperstacks for finished outputs")
    p.add_argument("directory")
    p.add_argument(
        "channels", nargs="*", help="channels or groups to export (default: all)"
    )
    p.add_argument(
        "--config", default=None, help="pyclm_config.toml for older HDF5 files"
    )
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("gui", help="open the live viewer on an experiment directory")
    p.add_argument("directory")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("control", help="open the control window: run, positions, files")
    p.add_argument("directory", nargs="?", default=None)
    p.add_argument("--config", default=None, help="path to pyclm_config.toml")
    p.add_argument("--dry", action="store_true", help="use the virtual microscope")
    p.set_defaults(func=cmd_control)

    p = sub.add_parser("new", help="create an experiment directory from a template")
    p.add_argument("directory")
    p.add_argument(
        "--template",
        default="closed-loop",
        choices=("open-loop", "closed-loop"),
        help="which template (default: closed-loop)",
    )
    p.add_argument(
        "--name", default="experiment", help="the experiment name (<name>.toml)"
    )
    p.set_defaults(func=cmd_new)
    return parser


# --------------------------------------------------------------- commands
def cmd_run(args) -> int:
    from .check import CheckFailed
    from .run_pyclm import run_pyclm

    try:
        run_pyclm(
            Path(args.directory),
            args.config,
            dry=args.dry,
            gui=args.gui,
            check=not args.no_check,
            force=args.force,
        )
    except CheckFailed as e:
        print(str(e))
        return 1
    return 0


def cmd_check(args) -> int:
    from .check import check_directory

    report = check_directory(
        Path(args.directory), args.config, mm_config=args.mm_config, dry=args.dry
    )
    print(report.text())
    return 0 if report.ok else 1


def cmd_preview(args) -> int:
    from .preview import preview

    result = preview(
        Path(args.directory),
        args.experiment,
        image=args.image,
        snap=args.snap,
        config_path=args.config,
        out_dir=args.out,
        t=args.t,
        pixel_size_um=args.pixel_size_um,
    )
    print(result.text())
    return 0


def cmd_export(args) -> int:
    from . import io as pyclm_io
    from .convert_hdf5s import find_affine_transform

    directory = Path(args.directory)
    wanted = set(args.channels)
    fallback = find_affine_transform(directory, args.config)
    written = []
    for path in pyclm_io.find_experiments(directory):
        with pyclm_io.open(path) as exp:
            groups = None
            if wanted:
                groups = [
                    name
                    for name, g in exp.groups.items()
                    if name in wanted
                    or any(c in wanted for c in g.channels)
                    or ("stim" in wanted and g.name == "stim_aq")
                ]
            affine = (
                exp.affine_transform if exp.affine_transform is not None else fallback
            )
            written += pyclm_io.export_imagej(exp, groups=groups, affine=affine)
    print(f"exported {len(written)} stack(s)")
    for path in written:
        print(f"  {path}")
    return 0


def cmd_gui(args) -> int:
    from .gui.gui_controller import main as gui_main

    return gui_main([args.directory])


def cmd_control(args) -> int:
    from .gui.control import main as control_main

    argv = [] if args.directory is None else [args.directory]
    if args.config:
        argv += ["--config", args.config]
    if args.dry:
        argv.append("--dry")
    return control_main(argv)


def cmd_new(args) -> int:
    from .templates import create

    written = create(Path(args.directory), args.template, args.name)
    print(f"created {args.template} experiment in {args.directory}")
    for path in written:
        print(f"  {path.name}")
    print("edit the files, then: pyclm check " + args.directory)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # the pre-Stage-5 form: pyclm <dir> [--dry] [--gui] [--config X]
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv = ["run", *argv]
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
