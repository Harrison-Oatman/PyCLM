"""
The ``pyclm`` command.

::

    pyclm run <dir> [--config FILE] [--dry] [--gui | --detached] [--no-check | --force]
    pyclm watch <dir> [--every-s S] [--once] [--images] [--on-flag CMD] [--then CMD]
    pyclm check <dir> [--config FILE] [--mm-config FILE] [--dry] [--fix]
    pyclm preview <dir> <experiment> (--image FILE | --snap) [--config FILE] [--out DIR] [--t N | --t A:B[:STEP]]
    pyclm export <dir> [channels ...] [--config FILE] [--movies [--fps N] [--scale X] [--split]]
    pyclm pack <dir> <destination> [--no-zip] [--checksums] [--force]
    pyclm pack --verify <copy> | --unpack <copy> <destination>
    pyclm stitch <dir> [--channel C] [--orientation O] [--every-s S]
    pyclm gui <dir>
    pyclm new <dir> [--template open-loop|closed-loop] [--name NAME] [--demo]

``pyclm <dir> [--dry] [--gui]`` (the form before Stage 5) still means ``run``.
Custom methods are loaded from ``methods`` in ``pyclm_config.toml`` and
from ``pyclm.methods`` entry points (:mod:`pyclm.methods`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMMANDS = (
    "run",
    "check",
    "preview",
    "export",
    "gui",
    "new",
    "watch",
    "pack",
    "stitch",
)


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
    p.add_argument(
        "--detached",
        action="store_true",
        help="check, then start the run in its own process and return; closing "
        "this terminal or agent does not stop it (output in <dir>/detached.log)",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no-check", action="store_true", help="skip the check")
    g.add_argument(
        "--force", action="store_true", help="run even if the check finds errors"
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="report problems in an experiment directory")
    p.add_argument("directory")
    p.add_argument(
        "--fix",
        action="store_true",
        help="first rewrite old pattern-method names and arguments in the "
        "experiment TOMLs (originals kept as .toml.bak)",
    )
    p.add_argument("--config", default=None, help="path to pyclm_config.toml")
    p.add_argument(
        "--mm-config", default=None, help="MicroManager .cfg to check presets against"
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="also run two timepoints on the virtual microscope",
    )
    p.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="unbinned camera pixel size, to compare a grid's spacing with camera_roi",
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
    src.add_argument(
        "--image",
        default=None,
        action="append",
        help="a TIF to use for every channel, or CHANNEL=TIF, repeated, one per "
        "channel (for a multi-channel segmentation: --image 545=a.tif --image 638=b.tif)",
    )
    src.add_argument("--snap", action="store_true", help="snap from the microscope")
    p.add_argument("--config", default=None, help="path to pyclm_config.toml")
    p.add_argument(
        "--out", default=None, help="output directory (default: <dir>/preview/<label>)"
    )
    p.add_argument(
        "--t",
        default="0",
        help="the timepoint to pretend it is, or start:stop[:step] for a series "
        "(0:240:5; the end is included)",
    )
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
    p.add_argument(
        "--movies",
        action="store_true",
        help="write MP4 preview movies to <dir>/movies instead of ImageJ stacks",
    )
    p.add_argument(
        "--fps", type=float, default=10.0, help="movie frames per second (10)"
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="movie size relative to the frames (1.0)",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="one movie per channel instead of side by side",
    )
    p.add_argument(
        "--no-pattern",
        action="store_true",
        help="leave the light pattern out of movies",
    )
    p.add_argument(
        "--no-outlines",
        action="store_true",
        help="leave segmentation outlines out of movies",
    )
    p.add_argument("--out", default=None, help="movie folder (default: <dir>/movies)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser(
        "watch",
        help="follow a running (or finished) experiment directory and flag problems",
    )
    p.add_argument("directory")
    p.add_argument(
        "--every-s", type=float, default=60.0, help="seconds between checks (60)"
    )
    p.add_argument(
        "--once", action="store_true", help="check once, print, and exit (1 if flags)"
    )
    p.add_argument(
        "--images",
        action="store_true",
        help="also flag brightness, saturation and focus changes from the newest frames",
    )
    p.add_argument(
        "--late-s",
        type=float,
        default=None,
        help="flag an experiment this many seconds behind (default: one interval)",
    )
    p.add_argument(
        "--stale-min",
        type=float,
        default=10.0,
        help="flag a status.json not rewritten for this many minutes (10)",
    )
    p.add_argument(
        "--on-flag",
        default=None,
        help="a command to run when new flags appear (gets the flags file as its last argument)",
    )
    p.add_argument(
        "--quiet-min",
        type=float,
        default=30.0,
        help="minutes before --on-flag runs again for the same flag (30)",
    )
    p.add_argument(
        "--then",
        default=None,
        help="a command to run once the run is done and its files have settled",
    )
    p.add_argument(
        "--out", default=None, help="where flags are written (default: <dir>/watch)"
    )
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser(
        "pack", help="copy a finished run as a few zip files, with a checked manifest"
    )
    p.add_argument("directory", help="the run, or the copy with --verify / --unpack")
    p.add_argument("destination", nargs="?", default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify", action="store_true", help="check a copy against its manifest"
    )
    mode.add_argument(
        "--unpack",
        action="store_true",
        help="check a packed copy, then unpack it into <destination>/<run>",
    )
    p.add_argument(
        "--no-zip",
        action="store_true",
        help="copy file by file instead of zipping folders",
    )
    p.add_argument(
        "--checksums",
        action="store_true",
        help="also hash every file (about twice as slow)",
    )
    p.add_argument(
        "--keep-exports",
        action="store_true",
        help="also copy the ImageJ exports (they duplicate the stores)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="copy even if the run does not say done or its files are still changing",
    )
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser(
        "stitch",
        help="draw every position's newest frame where the stage was, in one image",
    )
    p.add_argument("directory")
    p.add_argument(
        "--channel", default=None, help="channel to draw (default: the first)"
    )
    p.add_argument(
        "--orientation",
        default="identity",
        choices=(
            "identity",
            "flip_ud",
            "flip_lr",
            "rot90",
            "rot180",
            "rot270",
            "transpose",
        ),
        help="how each camera frame is turned before it is placed (identity)",
    )
    p.add_argument(
        "--every-s",
        type=float,
        default=None,
        help="redraw this often until the run is done (default: once)",
    )
    p.add_argument(
        "--max-px", type=int, default=3000, help="longest side of the image (3000)"
    )
    p.add_argument(
        "--out", default=None, help="output folder (default: <dir>/stitched)"
    )
    p.set_defaults(func=cmd_stitch)

    p = sub.add_parser("gui", help="open the live viewer on an experiment directory")
    p.add_argument("directory")
    p.set_defaults(func=cmd_gui)

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
    p.add_argument(
        "--demo",
        action="store_true",
        help="a demo that runs on the virtual microscope: two experiments and "
        "synthetic images (ignores --template and --name)",
    )
    p.set_defaults(func=cmd_new)
    return parser


# --------------------------------------------------------------- commands
def cmd_run(args) -> int:
    if getattr(args, "detached", False):
        return _run_detached(args)
    from .check import CheckFailed
    from .gui import GuiUnavailable
    from .lock import RunLocked
    from .methods import MethodLoadError
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
    except (GuiUnavailable, MethodLoadError, RunLocked) as e:
        print(f"error: {e}")
        return 1
    return 0


def _run_detached(args) -> int:
    from .check import check_directory
    from .detach import launch

    if args.gui:
        print(
            "error: --detached and --gui do not go together; open the viewer with pyclm gui"
        )
        return 1
    directory = Path(args.directory)
    if not args.dry:
        from .lock import RunLocked, holder, lock_path

        current = holder()
        if current is not None:
            print(f"error: {RunLocked(current, lock_path())}")
            return 1
    if not args.no_check:
        report = check_directory(directory, args.config)
        print(report.text())
        if report.errors and not args.force:
            print("not started: fix the errors above, or add --force")
            return 1
    started = launch(directory, args.config, dry=args.dry)
    print(started.text())
    return 0


def cmd_watch(args) -> int:
    from .watch import WatchSettings, watch

    settings = WatchSettings(
        every_s=args.every_s,
        images=args.images,
        late_s=args.late_s,
        stale_min=args.stale_min,
        on_flag=args.on_flag,
        quiet_min=args.quiet_min,
        then=args.then,
        out=Path(args.out) if args.out else None,
    )
    return watch(Path(args.directory), settings, once=args.once)


def cmd_check(args) -> int:
    from .check import check_directory

    if getattr(args, "fix", False):
        from .fix import fix_directory

        for result in fix_directory(Path(args.directory)):
            for change in result.changes:
                print(f"fixed   {result.path.name} {change}")
            for problem in result.problems:
                print(f"skipped {result.path.name} {problem}")
            if result.written:
                print(
                    f"wrote   {result.path.name} (was saved as {result.path.name}.bak)"
                )
        print()

    report = check_directory(
        Path(args.directory),
        args.config,
        mm_config=args.mm_config,
        dry=args.dry,
        pixel_size_um=args.pixel_size_um,
    )
    print(report.text())
    return 0 if report.ok else 1


def cmd_preview(args) -> int:
    from .methods import MethodLoadError
    from .preview import parse_timepoints, preview

    try:
        timepoints = parse_timepoints(args.t)
        image = _preview_images(args.image)
    except ValueError as e:
        print(f"error: {e}")
        return 1
    try:
        result = preview(
            Path(args.directory),
            args.experiment,
            image=image,
            snap=args.snap,
            config_path=args.config,
            out_dir=args.out,
            t=timepoints if len(timepoints) > 1 else timepoints[0],
            pixel_size_um=args.pixel_size_um,
        )
    except MethodLoadError as e:
        print(f"error: {e}")
        return 1
    print(result.text())
    return 0


def cmd_export(args) -> int:
    from . import io as pyclm_io
    from .convert_hdf5s import find_affine_transform

    directory = Path(args.directory)
    wanted = set(args.channels)
    if getattr(args, "movies", False):
        return _export_movies(directory, list(args.channels), args)
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


def _export_movies(directory: Path, channels: list[str], args) -> int:
    from . import io as pyclm_io
    from .io.movies import export_movies

    out = Path(args.out) if args.out else directory / "movies"
    written = []
    for path in pyclm_io.find_experiments(directory):
        with pyclm_io.open(path) as exp:
            made = export_movies(
                exp,
                out,
                channels=channels or None,
                scale=args.scale,
                fps=args.fps,
                pattern=not args.no_pattern,
                outlines=not args.no_outlines,
                split=args.split,
            )
            if not made:
                print(f"  {exp.name}: nothing imaged, no movie")
            written += made
    print(f"wrote {len(written)} movie(s)")
    for path in written:
        print(f"  {path}")
    return 0


def cmd_pack(args) -> int:
    from .pack import pack, unpack, verify

    if args.verify:
        problems = verify(Path(args.directory), hashes=args.checksums)
        if problems:
            print("COPY IS NOT GOOD:\n  " + "\n  ".join(problems[:50]))
            return 1
        print(f"copy verified: {args.directory}")
        return 0
    if args.destination is None:
        print("error: give a destination folder")
        return 2
    if args.unpack:
        return unpack(
            Path(args.directory), Path(args.destination), checksums=args.checksums
        )
    return pack(
        Path(args.directory),
        Path(args.destination),
        zip_folders=not args.no_zip,
        checksums=args.checksums,
        force=args.force,
        skip_exports=not args.keep_exports,
    )


def cmd_stitch(args) -> int:
    from .stitch import stitch

    return stitch(
        Path(args.directory),
        every_s=args.every_s,
        out=Path(args.out) if args.out else None,
        channel=args.channel,
        orientation=args.orientation,
        max_px=args.max_px,
    )


def cmd_gui(args) -> int:
    from .gui import GuiUnavailable, require_gui

    try:
        require_gui()
    except GuiUnavailable as e:
        print(f"error: {e}")
        return 1
    from .gui.gui_controller import main as gui_main

    return gui_main([args.directory])


def cmd_new(args) -> int:
    if args.demo:
        from .demo import create_demo

        written = create_demo(Path(args.directory))
        print(f"created the demo in {args.directory}")
        for path in written:
            print(f"  {path.name}")
        print(f"try: pyclm run {args.directory} --dry   (add --gui to watch)")
        return 0
    from .templates import create

    written = create(Path(args.directory), args.template, args.name)
    print(f"created {args.template} experiment in {args.directory}")
    for path in written:
        print(f"  {path.name}")
    print("edit the files, then: pyclm check " + args.directory)
    return 0


def _preview_images(values: list[str] | None):
    """``--image`` values: one path for every channel, or ``CHANNEL=path`` per channel."""
    if not values:
        return None
    pairs = [v.split("=", 1) for v in values if "=" in v]
    if not pairs:
        if len(values) > 1:
            raise ValueError("several --image values must each be CHANNEL=TIF")
        return values[0]
    if len(pairs) != len(values):
        raise ValueError("mix of --image TIF and --image CHANNEL=TIF; use one form")
    return {channel: path for channel, path in pairs}


def main(argv: list[str] | None = None) -> int:
    from .console import utf8_console

    utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    # the pre-Stage-5 form: pyclm <dir> [--dry] [--gui] [--config X]
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv = ["run", *argv]
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
