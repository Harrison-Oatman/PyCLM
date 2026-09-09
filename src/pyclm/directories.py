import json
import logging
from pathlib import Path
from xml.etree import ElementTree

import yaml

from .core import ExperimentSchedule
from .core.experiments import MicroscopePosition, PositionWithAutoFocus
from .core.virtual_microscope.simulated_source import TimeSeriesImageSource

logger = logging.getLogger(__name__)


def experiment_from_toml(toml_path, name="SampleExperiment"):
    """
    The ``Experiment`` described by one experiment TOML, for position ``name``.
    Validated against :class:`pyclm.schema.ExperimentConfig`; raises
    :class:`pyclm.schema.ConfigError` listing every problem in the file.
    """
    from .schema import ExperimentConfig

    return ExperimentConfig.from_file(toml_path).to_experiment(name)


def read_schedule(toml_path):
    """The timing of ``schedule.toml`` as ``ExperimentSchedule`` keyword arguments."""
    from .schema import ScheduleConfig

    return ScheduleConfig.from_file(toml_path).timing_kwargs()


def positions_from_pos(fp) -> list[MicroscopePosition]:
    """
    Parse a MicroManager PositionList.pos file (JSON Property Map v2).

    The XY stage is identified by the ``DefaultXYStage`` field in each position
    entry.  The Z stage is the first single-axis device whose name is not the
    XY stage.  All remaining single-axis devices are stored as optional extras
    keyed by their device name (e.g. ``{"PFSOffset": 11122.0}``).
    """
    with open(fp) as f:
        data = json.load(f)

    positions = []

    for pos_data in data["map"]["StagePositions"]["array"]:
        label = str(pos_data["Label"]["scalar"]).replace("-", ".")
        xy_stage = pos_data["DefaultXYStage"]["scalar"]
        z_stage = pos_data["DefaultZStage"]["scalar"]

        x = y = z = None
        extras: dict = {}

        for dev_entry in pos_data["DevicePositions"]["array"]:
            device = dev_entry["Device"]["scalar"]
            values = [float(v) for v in dev_entry["Position_um"]["array"]]

            if device == xy_stage:
                x, y = values[0], values[1]
            elif device == z_stage:
                z = values[0]
            elif z_stage == "" and z is None and len(values) == 1:
                # First non-XY single-axis device is the Z stage
                z = values[0]
            else:
                extras[device] = values[0] if len(values) == 1 else values

        if x is None or y is None:
            raise ValueError(f"Position '{label}' in {fp} has no XY stage entry.")
        if z is None:
            raise ValueError(f"Position '{label}' in {fp} has no Z stage entry.")

        positions.append(MicroscopePosition(x=x, y=y, z=z, label=label, extras=extras))

    return positions


def positions_from_xml(fp):
    tree = ElementTree.parse(fp)
    root = tree.getroot()

    positions = []

    for node in root[0]:
        if node.attrib["runtype"] != "NDSetupMultipointListItem":
            continue

        nv = {
            n.tag: n.attrib["value"]
            for n in node
            if n.attrib.get("value", None) is not None
        }

        for tag in ["dXPosition", "dYPosition", "dZPosition", "dPFSOffset"]:
            val = nv.get(tag)
            if val is not None:
                val = float(val)

            nv[tag] = val

        if (nv["dPFSOffset"] is not None) and (nv["dPFSOffset"] < 0):
            nv["dPFSOffset"] = None

        positions.append(
            PositionWithAutoFocus(
                x=nv["dXPosition"],
                y=nv["dYPosition"],
                z=nv["dZPosition"],
                autofocus_offset=nv["dPFSOffset"],
                label=nv["strName"],
            )
        )

    return positions


def schedule_from_directory(experiment_dir: Path):
    tomls = experiment_dir.glob("*.toml")
    tomls = {f.stem: str(f) for f in tomls}

    pos_path = experiment_dir / "PositionList.pos"
    xml_path = experiment_dir / "multipoints.xml"

    if pos_path.exists():
        pos_list = positions_from_pos(str(pos_path))
    elif xml_path.exists():
        pos_list = positions_from_xml(str(xml_path))
    else:
        raise FileNotFoundError(
            f"No position list found in {experiment_dir}. "
            "Expected 'PositionList.pos' or 'multipoints.xml'."
        )

    positions = {}
    experiments = {}

    for position in pos_list:
        name: str = position.label

        exp_stem = name.split(".")[0]

        experiment_path = tomls.get(exp_stem)

        if experiment_path is None:
            logger.warning(
                f"position '{name}': no experiment toml named '{exp_stem}' in "
                f"{sorted(tomls)}; position skipped"
            )
            continue

        positions[name] = position
        experiments[name] = experiment_from_toml(experiment_path, name)

    timing = read_schedule(str(experiment_dir / "schedule.toml"))

    schedule = ExperimentSchedule(experiments, positions, **timing)

    return schedule


# ---------------------------------------------------------------------------
# Dry-run schedule discovery
# ---------------------------------------------------------------------------


def _dry_schedule_from_yml(
    experiment_dir: Path,
    yml_path: Path,
    tomls: dict[str, str],
    timing: dict,
) -> tuple[ExperimentSchedule, TimeSeriesImageSource]:
    """Build a dry-run schedule from an explicit dry_run.yml file.

    Each entry must have ``name`` (position label) and ``source`` (TIF path
    relative to the experiment directory).  Optional ``x``, ``y``, ``z``
    fields override the placeholder coordinates.
    """
    with open(yml_path) as f:
        data = yaml.safe_load(f)

    entries = data.get("positions", [])
    if not entries:
        raise ValueError(f"{yml_path} contains no positions")

    positions: dict[str, MicroscopePosition] = {}
    experiments: dict[str, object] = {}
    pos_to_tif: dict[tuple[float, float], Path] = {}

    for i, entry in enumerate(entries):
        name: str = entry["name"]
        tif_path = experiment_dir / entry["source"]

        if not tif_path.exists():
            raise FileNotFoundError(
                f"dry_run.yml source '{entry['source']}' not found: {tif_path}"
            )

        exp_stem = name.split(".")[0]
        experiment_path = tomls.get(exp_stem)
        if experiment_path is None:
            raise FileNotFoundError(
                f"No TOML found for stem '{exp_stem}' (from position '{name}'). "
                f"Available TOMLs: {list(tomls.keys())}"
            )

        x = float(entry.get("x", i * 10000))
        y = float(entry.get("y", 0.0))
        z = float(entry.get("z", 0.0))

        pos = MicroscopePosition(x=x, y=y, z=z, label=name)
        positions[name] = pos
        experiments[name] = experiment_from_toml(experiment_path, name)
        pos_to_tif[(x, y)] = tif_path

    schedule = ExperimentSchedule(experiments, positions, **timing)
    image_source = TimeSeriesImageSource.from_mapping(pos_to_tif, loop=True)
    return schedule, image_source


def _dry_schedule_from_position_list(
    experiment_dir: Path,
    tomls: dict[str, str],
    timing: dict,
) -> tuple[ExperimentSchedule, TimeSeriesImageSource]:
    """Build a dry-run schedule from an existing position list (pos/xml).

    For each position, TIF matching order:
      1. ``{label}.tif`` — exact label match
      2. ``{experiment_stem}.tif`` — stem match (e.g. bar10.tif for bar10.pos1)
      3. Round-robin from all TIFs in the directory
    """
    pos_path = experiment_dir / "PositionList.pos"
    xml_path = experiment_dir / "multipoints.xml"

    if pos_path.exists():
        pos_list = positions_from_pos(str(pos_path))
    else:
        pos_list = positions_from_xml(str(xml_path))

    available_tifs = sorted(experiment_dir.glob("*.tif"))
    if not available_tifs:
        raise FileNotFoundError(f"No TIF files found in {experiment_dir} for dry run")

    tif_by_name = {t.stem: t for t in available_tifs}

    positions: dict[str, MicroscopePosition] = {}
    experiments: dict[str, object] = {}
    pos_to_tif: dict[tuple[float, float], Path] = {}

    for position in pos_list:
        name: str = position.label
        exp_stem = name.split(".")[0]

        experiment_path = tomls.get(exp_stem)
        if experiment_path is None:
            continue

        if name in tif_by_name:
            tif_path = tif_by_name[name]
        elif exp_stem in tif_by_name:
            tif_path = tif_by_name[exp_stem]
        else:
            tif_path = available_tifs[len(positions) % len(available_tifs)]

        pos_to_tif[(position.x, position.y)] = tif_path
        positions[name] = position
        experiments[name] = experiment_from_toml(experiment_path, name)

    if not positions:
        raise FileNotFoundError(
            f"No positions in the position list matched any TOML files. "
            f"Available TOMLs: {list(tomls.keys())}"
        )

    schedule = ExperimentSchedule(experiments, positions, **timing)
    image_source = TimeSeriesImageSource.from_mapping(pos_to_tif, loop=True)
    return schedule, image_source


def _dry_schedule_from_tifs(
    experiment_dir: Path,
    tomls: dict[str, str],
    timing: dict,
) -> tuple[ExperimentSchedule, TimeSeriesImageSource]:
    """Build a dry-run schedule from TIF filenames alone.

    Each TIF stem becomes a position label; the first dot-separated token
    is matched to a TOML (e.g. ``fast.00.tif`` → label ``fast.00``,
    experiment ``fast.toml``).  TIFs with no matching TOML are silently
    skipped.  Placeholder coordinates are assigned sequentially.
    """
    tif_files = sorted(experiment_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {experiment_dir} for dry run")

    positions: dict[str, MicroscopePosition] = {}
    experiments: dict[str, object] = {}
    pos_to_tif: dict[tuple[float, float], Path] = {}

    for i, tif_path in enumerate(tif_files):
        label = tif_path.stem
        exp_stem = label.split(".")[0]

        experiment_path = tomls.get(exp_stem)
        if experiment_path is None:
            continue

        x, y = float(i * 10000), 0.0
        pos = MicroscopePosition(x=x, y=y, z=0.0, label=label)
        positions[label] = pos
        experiments[label] = experiment_from_toml(experiment_path, label)
        pos_to_tif[(x, y)] = tif_path

    if not positions:
        raise FileNotFoundError(
            f"No TIF files in {experiment_dir} matched any TOML files. "
            f"TIF stems: {[t.stem for t in tif_files]}, "
            f"available TOMLs: {list(tomls.keys())}"
        )

    schedule = ExperimentSchedule(experiments, positions, **timing)
    image_source = TimeSeriesImageSource.from_mapping(pos_to_tif, loop=True)
    return schedule, image_source


def dry_schedule_from_directory(
    experiment_dir: Path,
) -> tuple[ExperimentSchedule, TimeSeriesImageSource]:
    """Discover experiments and image sources for a dry run.

    Priority order:
      1. ``dry_run.yml`` — explicit position-to-TIF mapping
      2. ``PositionList.pos`` or ``multipoints.xml`` — real positions,
         TIFs matched by label/stem/round-robin
      3. TIF filenames in the directory — position labels inferred from stems
    """
    experiment_dir = Path(experiment_dir)
    tomls = {f.stem: str(f) for f in experiment_dir.glob("*.toml")}
    timing = read_schedule(str(experiment_dir / "schedule.toml"))

    yml_path = experiment_dir / "dry_run.yml"
    pos_path = experiment_dir / "PositionList.pos"
    xml_path = experiment_dir / "multipoints.xml"

    if yml_path.exists():
        return _dry_schedule_from_yml(experiment_dir, yml_path, tomls, timing)

    if pos_path.exists() or xml_path.exists():
        return _dry_schedule_from_position_list(experiment_dir, tomls, timing)

    return _dry_schedule_from_tifs(experiment_dir, tomls, timing)


def write_position_list(path, positions, xy_stage="XYStage", z_stage="ZDrive"):
    """
    Write positions as a MicroManager ``PositionList.pos`` (property map v2),
    the format :func:`positions_from_pos` reads and MicroManager imports.
    ``extras`` of a position (e.g. ``PFSOffset``) become single-axis devices.
    """
    path = Path(path)
    entries = []
    for i, pos in enumerate(positions):
        devices = [
            {
                "Device": {"type": "STRING", "scalar": xy_stage},
                "Position_um": {
                    "type": "DOUBLE",
                    "array": [float(pos.x), float(pos.y)],
                },
            },
            {
                "Device": {"type": "STRING", "scalar": z_stage},
                "Position_um": {"type": "DOUBLE", "array": [float(pos.z)]},
            },
        ]
        for device, value in (getattr(pos, "extras", {}) or {}).items():
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            devices.append(
                {
                    "Device": {"type": "STRING", "scalar": str(device)},
                    "Position_um": {
                        "type": "DOUBLE",
                        "array": [float(v) for v in values],
                    },
                }
            )
        entries.append(
            {
                "DefaultXYStage": {"type": "STRING", "scalar": xy_stage},
                "DefaultZStage": {"type": "STRING", "scalar": z_stage},
                "DevicePositions": {"type": "PROPERTY_MAP", "array": devices},
                "GridCol": {"type": "INTEGER", "scalar": 0},
                "GridRow": {"type": "INTEGER", "scalar": i},
                "Label": {"type": "STRING", "scalar": str(pos.label)},
                "Properties": {"type": "PROPERTY_MAP", "scalar": {}},
            }
        )
    document = {
        "encoding": "UTF-8",
        "format": "Micro-Manager Property Map",
        "major_version": 2,
        "minor_version": 0,
        "map": {"StagePositions": {"type": "PROPERTY_MAP", "array": entries}},
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path
