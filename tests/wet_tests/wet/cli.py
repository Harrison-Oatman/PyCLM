"""``wet``: prepare, run, verify and report the wet-test items."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import toml

HERE = Path(__file__).resolve().parent.parent
EXPERIMENTS = HERE / "experiments"
RESULTS = HERE / "results"
TEMPLATE = HERE / "config_template.toml"
OUTPUT_SUFFIXES = (".zarr", ".hdf5")
OUTPUT_FILES = (
    "frames.parquet",
    "frames.csv",
    "tracks.parquet",
    "tracks.csv",
    "events.parquet",
    "events.csv",
    "status.json",
    "plan.useq.yaml",
    "log.log",
    "all_layers.txt",
)
OUTPUT_DIRS = ("commands", "preview", "imagej")


def items() -> dict[str, Path]:
    return {p.name[:2]: p for p in sorted(EXPERIMENTS.iterdir()) if p.is_dir()}


def resolve(item: str) -> Path:
    table = items()
    key = item[:2]
    if key not in table:
        raise SystemExit(f"no item {item!r}; run `wet list`")
    return table[key]


def pyclm_commit() -> str:
    try:
        import pyclm

        repo = Path(pyclm.__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ prepare
def prepare(directory: Path, quiet: bool = False) -> Path:
    """Copy the config template into the directory, apply overrides.toml, run the check."""
    if not TEMPLATE.exists():
        raise SystemExit(f"{TEMPLATE} is missing")
    config = tomllib.loads(TEMPLATE.read_text(encoding="utf-8"))
    # JSON, not TOML: every top-level .toml in an experiment directory is an experiment
    overrides = directory / "overrides.json"
    if overrides.exists():
        _merge(config, json.loads(overrides.read_text(encoding="utf-8")))
    target = directory / "pyclm_config.toml"
    target.write_text(toml.dumps(config), encoding="utf-8")
    if not quiet:
        print(f"wrote {target}")
        guide = directory / "guide.md"
        if guide.exists():
            print(f"read {guide} for what this item needs live")
    return target


def _merge(base: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def check(directory: Path) -> bool:
    from pyclm.check import check_directory

    from .methods import PATTERN_METHODS

    report = check_directory(directory, pattern_methods=PATTERN_METHODS)
    print(report.text())
    return report.ok


# ------------------------------------------------------------------ run
def run(directory: Path, dry: bool = False, force: bool = False) -> None:
    from pyclm import run_pyclm

    from .methods import PATTERN_METHODS

    prepare(directory)
    run_pyclm(
        directory,
        dry=dry,
        pattern_methods=PATTERN_METHODS,
        gui=False,
        force=force,
    )


def clean(directory: Path) -> None:
    # the ImageJ export writes <label>_<group>.tif next to each store: remove
    # those with the store (a dry run would take them for position TIFs)
    for store in list(directory.iterdir()):
        if store.suffix in OUTPUT_SUFFIXES:
            for tif in directory.glob(f"{store.stem}_*.tif"):
                tif.unlink()
    for p in directory.iterdir():
        if p.suffix in OUTPUT_SUFFIXES or p.name in OUTPUT_FILES:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        elif p.is_dir() and p.name in OUTPUT_DIRS:
            shutil.rmtree(p)
    print(f"cleaned {directory.name}")


# ------------------------------------------------------------------ verify + report
def verify(directory: Path) -> list[tuple[str, bool, str]]:
    from . import verify as checks

    fn = getattr(checks, f"verify_{directory.name[:2]}", None)
    if fn is None:
        return [
            (
                "manual",
                True,
                f"no automatic verification; follow {directory.name}/guide.md",
            )
        ]
    try:
        return fn(directory)
    except Exception as e:
        return [("verification", False, f"crashed: {e!r}")]


def write_result(directory: Path, rows: list[tuple[str, bool, str]]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{directory.name}.md"
    ok = all(r[1] for r in rows)
    lines = [
        f"# {directory.name}: {'PASS' if ok else 'FAIL'}",
        "",
        f"- date: {datetime.datetime.now().isoformat(timespec='minutes')}",
        f"- pyclm commit: {pyclm_commit()}",
        f"- directory: {directory}",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for name, passed, detail in rows:
        lines.append(f"| {name} | {'pass' if passed else 'FAIL'} | {detail} |")
    lines += ["", "Notes (fill in by hand: anything that surprised you):", "", ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out}")
    return out


def report() -> Path:
    RESULTS.mkdir(exist_ok=True)
    parts = [
        f"# Wet-test report ({datetime.date.today().isoformat()}, pyclm {pyclm_commit()})",
        "",
    ]
    for directory in items().values():
        result = RESULTS / f"{directory.name}.md"
        if result.exists():
            first = result.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
            parts.append(f"- {first}")
        else:
            parts.append(f"- {directory.name}: not run")
    out = RESULTS / "REPORT.md"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("\n".join(parts))
    return out


# ------------------------------------------------------------------ commands (item 08)
def send_commands(directory: Path, experiment: str | None) -> None:
    """The command sequence of item 08, sent to a run in progress in ``directory``."""
    import time

    from pyclm.commands import write_command

    status_path = directory / "status.json"
    if not status_path.exists():
        raise SystemExit(
            "no status.json: start `wet run 08` first, in another terminal"
        )
    status = json.loads(status_path.read_text())
    experiment = experiment or next(iter(status.get("experiments", {})), None)
    if experiment is None:
        raise SystemExit(
            "the run has not acknowledged an experiment yet; wait a timepoint"
        )
    interval = _interval(directory)

    def send(cmd: dict, wait: float):
        path = write_command(directory, cmd)
        print(f"sent {cmd}  ({path.name}); waiting {wait:.0f}s")
        time.sleep(wait)

    send({"command": "pause"}, 2.5 * interval)
    send({"command": "resume"}, 1.5 * interval)
    send(
        {
            "command": "set_exposure",
            "experiment": experiment,
            "channel": "stimulation",
            "ms": 80,
        },
        interval,
    )
    send(
        {
            "command": "set_property",
            "experiment": experiment,
            "channel": "stimulation",
            "device": "Sola",
            "property": "Power",
            "value": 40,
        },
        interval,
    )
    send(
        {
            "command": "set_position",
            "experiment": experiment,
            "z": _z_plus(directory, experiment, 2.0),
        },
        interval,
    )
    send(
        {
            "command": "set_pattern",
            "experiment": experiment,
            "parameters": {"bar_speed": 2.0},
        },
        2 * interval,
    )
    send({"command": "stop_run"}, 0)
    print("done; when the run has finished: `wet verify 08`")


def _z_plus(directory: Path, experiment: str, dz: float) -> float:
    """The experiment's last recorded z plus ``dz`` (an absolute target for set_position)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(directory / "frames.parquet").to_pylist()
    zs = [
        r["z"] for r in rows if r["experiment"] == experiment and r.get("z") is not None
    ]
    return float(zs[-1]) + dz if zs else dz


def _interval(directory: Path) -> float:
    from pyclm.directories import find_schedule
    from pyclm.schema import ScheduleConfig

    path = find_schedule(directory)
    return (
        float(ScheduleConfig.from_file(path).timing.interval_seconds) if path else 10.0
    )


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wet", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the items and their directories")
    for name, help_ in (
        (
            "prepare",
            "copy the config into the item, apply overrides.json, run pyclm check",
        ),
        ("run", "prepare, run, verify, write the result"),
        ("verify", "verify an item that has run and write the result"),
        ("clean", "remove an item's outputs so it can run again"),
        ("commands", "item 08: send the command sequence to a run in progress"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("item", help="the two-digit item number, e.g. 02")
        if name == "run":
            p.add_argument("--dry", action="store_true", help="virtual microscope")
            p.add_argument(
                "--force", action="store_true", help="run despite check errors"
            )
        if name == "commands":
            p.add_argument("--experiment", default=None)
    sub.add_parser("report", help="collate results/*.md into results/REPORT.md")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for key, path in items().items():
            print(f"{key}  {path.name}")
        return 0
    if args.cmd == "report":
        report()
        return 0
    directory = resolve(args.item)
    if args.cmd == "prepare":
        prepare(directory)
        return 0 if check(directory) else 1
    if args.cmd == "clean":
        clean(directory)
        return 0
    if args.cmd == "commands":
        send_commands(directory, args.experiment)
        return 0
    if args.cmd == "run":
        run(directory, dry=args.dry, force=args.force)
    rows = verify(directory)
    write_result(directory, rows)
    return 0 if all(r[1] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
