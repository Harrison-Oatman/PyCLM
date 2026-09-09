"""
``pyclm check``: read an experiment directory and report what is wrong
with it before the microscope is touched.

Every finding names a file and, where it applies, a table or key, and is
an error (the run cannot start), a warning (it can, but something looks
unintended) or an info line (a fact worth knowing: frames per experiment,
what was not checked). ``pyclm run`` runs the same check first.

See docs/stage5-schema-setup-design.md §4.
"""

from __future__ import annotations

import difflib
import inspect
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .core.patterns import known_models
from .core.plan import AcquisitionPlan
from .core.segmentation_process import SegmentationProcess
from .core.tracking import known_tracking_methods
from .mmconfig import MMConfig
from .schema import (
    ConfigError,
    ExperimentConfig,
    PyclmConfig,
    ScheduleConfig,
)

logger = logging.getLogger(__name__)

CONFIG_NAME = "pyclm_config.toml"
SCHEDULE_NAME = "schedule.toml"
OUTPUT_SUFFIXES = (".zarr", ".hdf5")


class CheckFailed(RuntimeError):
    """``pyclm run`` stopped because the check found errors (see ``report``)."""

    def __init__(self, report: CheckReport):
        self.report = report
        super().__init__(
            f"{report.summary()} in {report.directory}; fix them or run with --force"
        )


@dataclass
class Finding:
    level: str  # "error" | "warning" | "info"
    file: str
    where: str
    message: str

    def __str__(self) -> str:
        where = f" {self.where}" if self.where else ""
        return f"{self.level.upper():8}{self.file}{where}: {self.message}"


@dataclass
class CheckReport:
    directory: Path
    findings: list[Finding] = field(default_factory=list)

    def add(self, level: str, file: str, where: str, message: str) -> Finding:
        f = Finding(level, file, where, message)
        self.findings.append(f)
        return f

    def error(self, file, where, message):
        return self.add("error", file, where, message)

    def warning(self, file, where, message):
        return self.add("warning", file, where, message)

    def info(self, file, where, message):
        return self.add("info", file, where, message)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        n_err, n_warn = len(self.errors), len(self.warnings)
        if not n_err and not n_warn:
            return "no problems found"
        parts = []
        if n_err:
            parts.append(f"{n_err} error{'s' if n_err != 1 else ''}")
        if n_warn:
            parts.append(f"{n_warn} warning{'s' if n_warn != 1 else ''}")
        return ", ".join(parts)

    def text(self) -> str:
        lines = [f"pyclm check {self.directory}"]
        lines += [f"  {f}" for f in self.findings]
        lines.append(f"  {self.summary()}")
        return "\n".join(lines)


# --------------------------------------------------------------- helpers
def find_pyclm_config(directory: Path, config_path=None) -> Path | None:
    """``pyclm_config.toml``: the given path, else the directory's, else the working directory's."""
    if config_path is not None:
        return Path(config_path)
    for candidate in (Path(directory) / CONFIG_NAME, Path.cwd() / CONFIG_NAME):
        if candidate.exists():
            return candidate
    return None


def method_registries(
    pattern_methods=None, segmentation_methods=None, tracking_methods=None
):
    patterns = dict(known_models)
    patterns.update(pattern_methods or {})
    segmentations = dict(SegmentationProcess.default_models)
    segmentations.update(segmentation_methods or {})
    trackers = dict(known_tracking_methods)
    trackers.update(tracking_methods or {})
    return patterns, segmentations, trackers


# constructor parameters PyCLM supplies itself, per kind of method
SUPPLIED = {
    "pattern": {"self", "experiment_name", "camera_properties"},
    "segmentation": {"self", "experiment_name"},
    "tracking": {"self", "experiment_name", "channel"},
}


def _named_parameters(cls: type, supplied: set) -> tuple[dict, bool]:
    """``cls.__init__``'s named parameters (minus the ones PyCLM supplies) and whether it takes **kwargs."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}, True
    named = {}
    var_kw = False
    for name, param in sig.parameters.items():
        if name in supplied:
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            var_kw = True
        elif param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            named[name] = param
    return named, var_kw


def _all_subclasses(cls: type) -> list[type]:
    out = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out += _all_subclasses(sub)
    return out


def check_kwargs(
    cls: type, kwargs: dict, kind: str = "pattern", method_name: str | None = None
) -> list[str]:
    """
    Problems with ``kwargs`` against ``cls.__init__``: unknown names (with a
    did-you-mean) and missing required ones. ``kind`` says which parameters
    PyCLM supplies itself; ``method_name`` is the TOML name for messages.
    """
    supplied = SUPPLIED.get(kind, SUPPLIED["pattern"])
    label = method_name or getattr(cls, "name", cls.__name__)
    named, var_kw = _named_parameters(cls, supplied)
    dispatching = False
    if not named and "__new__" in vars(cls):
        # a base that picks a subclass in __new__ (bar, wave): accept what any subclass accepts
        dispatching = True
        for sub in _all_subclasses(cls):
            sub_named, _ = _named_parameters(sub, supplied)
            named.update(sub_named)
    problems = []
    for key in kwargs:
        if key in named:
            continue
        hint = ""
        close = difflib.get_close_matches(key, list(named), n=1)
        if close:
            hint = f" (did you mean {close[0]!r}?)"
        accepted = ", ".join(sorted(named)) or "none"
        problems.append(
            f"unknown argument {key!r} for method '{label}'{hint}; accepted: {accepted}"
            + (
                ""
                if not var_kw
                else " (extra keys are silently ignored by this method)"
            )
        )
    if not dispatching:
        for name, param in named.items():
            if param.default is inspect.Parameter.empty and name not in kwargs:
                problems.append(f"method '{label}' requires the argument {name!r}")
    return problems


def _load_positions(directory: Path):
    from .directories import positions_from_pos, positions_from_xml

    pos_path = directory / "PositionList.pos"
    xml_path = directory / "multipoints.xml"
    if pos_path.exists():
        return pos_path.name, positions_from_pos(str(pos_path))
    if xml_path.exists():
        return xml_path.name, positions_from_xml(str(xml_path))
    return None, None


# ------------------------------------------------------------------ check
def check_directory(
    directory,
    config_path=None,
    mm_config=None,
    pattern_methods=None,
    segmentation_methods=None,
    tracking_methods=None,
    dry: bool = False,
) -> CheckReport:
    """Check an experiment directory; see the module docstring for what is covered."""
    directory = Path(directory)
    report = CheckReport(directory)
    if not directory.is_dir():
        report.error(str(directory), "", "is not a directory")
        return report

    patterns, segmentations, trackers = method_registries(
        pattern_methods, segmentation_methods, tracking_methods
    )

    # 1. pyclm_config.toml
    config = None
    cfg_path = find_pyclm_config(directory, config_path)
    if cfg_path is None:
        report.error(
            CONFIG_NAME, "", "not found in the directory or the working directory"
        )
    else:
        try:
            config = PyclmConfig.from_file(cfg_path)
            report.info(cfg_path.name, "", f"ok ({cfg_path})")
        except ConfigError as e:
            for problem in e.problems:
                report.error(cfg_path.name, "", problem)

    # 2. schedule.toml
    schedule = None
    schedule_path = directory / SCHEDULE_NAME
    try:
        schedule = ScheduleConfig.from_file(schedule_path)
        t = schedule.timing
        report.info(
            SCHEDULE_NAME,
            "",
            f"{t.steps} timepoints every {t.interval_seconds:g} s "
            f"({t.steps * t.interval_seconds / 60:.1f} min)",
        )
    except ConfigError as e:
        for problem in e.problems:
            report.error(SCHEDULE_NAME, "", problem)

    # 3. experiment TOMLs
    configs: dict[str, ExperimentConfig] = {}
    toml_paths = sorted(
        p
        for p in directory.glob("*.toml")
        if p.name not in (CONFIG_NAME, SCHEDULE_NAME)
    )
    if not toml_paths:
        report.error(str(directory.name), "", "no experiment TOML files found")
    for path in toml_paths:
        try:
            configs[path.stem] = ExperimentConfig.from_file(path)
        except ConfigError as e:
            for problem in e.problems:
                report.error(path.name, "", problem)

    # 4. positions
    labels: dict[str, str] = {}  # label -> stem
    positions = None
    try:
        list_name, positions = _load_positions(directory)
    except Exception as e:
        list_name, positions = None, None
        report.error("position list", "", f"could not be read: {e}")
    if positions is not None:
        seen = set()
        for pos in positions:
            label = pos.label
            if label in seen:
                report.error(list_name, label, "duplicate position label")
            seen.add(label)
            stem = label.split(".")[0]
            if stem in configs:
                labels[label] = stem
            elif stem in {p.stem for p in toml_paths}:
                labels[label] = stem  # its TOML failed validation; reported above
            else:
                report.error(
                    list_name,
                    label,
                    f"no experiment file '{stem}.toml' for this position "
                    f"(files: {', '.join(p.stem for p in toml_paths) or 'none'})",
                )
        used = {stem for stem in labels.values()}
        for path in toml_paths:
            if path.stem not in used:
                report.warning(path.name, "", "not used by any position")
        report.info(
            list_name, "", f"{len(labels)} position(s): {', '.join(labels) or 'none'}"
        )
    else:
        dry_sources = list(directory.glob("*.tif")) + list(
            directory.glob("dry_run.yml")
        )
        if dry_sources:
            report.info(
                "positions",
                "",
                "no PositionList.pos or multipoints.xml; a dry run will use the TIF files",
            )
        else:
            report.error(
                "positions",
                "",
                "no PositionList.pos or multipoints.xml in the directory",
            )

    # 5 + 6. methods and requirements, per TOML
    requirements_by_stem: dict[str, list] = {}
    methods_by_stem: dict[str, object] = {}
    for stem, cfg in configs.items():
        file = f"{stem}.toml"
        pattern_cls = patterns.get(cfg.pattern.method)
        if pattern_cls is None:
            report.error(
                file,
                "[pattern]",
                f"unknown pattern method '{cfg.pattern.method}' "
                f"(known: {', '.join(sorted(patterns))})",
            )
        else:
            for problem in check_kwargs(
                pattern_cls, cfg.pattern.kwargs, "pattern", cfg.pattern.method
            ):
                report.error(file, "[pattern]", problem)

        for name, table in (
            [("segmentation", cfg.segmentation)] if cfg.segmentation else []
        ) + [(f"segmentation.{n}", t) for n, t in cfg.segmentations.items()]:
            seg_cls = segmentations.get(table.method)
            if table.method == "none":
                continue
            if seg_cls is None:
                report.error(
                    file,
                    f"[{name}]",
                    f"unknown segmentation method '{table.method}' "
                    f"(known: {', '.join(sorted(segmentations))})",
                )
            else:
                for problem in check_kwargs(
                    seg_cls, table.kwargs, "segmentation", table.method
                ):
                    report.error(file, f"[{name}]", problem)

        if cfg.tracking is not None:
            tr_cls = trackers.get(cfg.tracking.method)
            if tr_cls is None:
                report.error(
                    file,
                    "[tracking]",
                    f"unknown tracking method '{cfg.tracking.method}' "
                    f"(known: {', '.join(sorted(trackers))})",
                )
            else:
                for problem in check_kwargs(
                    tr_cls, cfg.tracking.kwargs, "tracking", cfg.tracking.method
                ):
                    report.error(file, "[tracking]", problem)

        if pattern_cls is None:
            continue
        try:
            method = pattern_cls(**cfg.pattern.kwargs)
        except Exception as e:
            report.error(
                file, "[pattern]", f"constructing '{cfg.pattern.method}' failed: {e!r}"
            )
            continue
        methods_by_stem[stem] = method
        experiment = cfg.to_experiment(f"{stem}.check")
        wanted = list(getattr(method, "_requirements_list", []))
        needs_default_seg = needs_tracks = False
        named_wanted: set[str] = set()
        for channel, _raw, seg_names, tracks, _history in wanted:
            if channel not in cfg.channels.presets:
                report.error(
                    file,
                    "[pattern]",
                    f"method '{cfg.pattern.method}' needs channel '{channel}', "
                    f"which is not in [channels] presets {cfg.channels.presets}",
                )
            for seg in seg_names:
                if seg == "segmentation":
                    needs_default_seg = True
                else:
                    named_wanted.add(seg)
            needs_tracks = needs_tracks or bool(tracks)
        if getattr(method, "_stim_requested", False):
            if getattr(method, "_stim_request_seg", False):
                needs_default_seg = True
            if cfg.stimulation.exposure <= 0:
                report.error(
                    file,
                    "[pattern]",
                    f"method '{cfg.pattern.method}' requests the stimulation frame but "
                    "[stimulation] exposure is 0",
                )
        if needs_default_seg and cfg.segmentation is None:
            report.error(
                file,
                "[segmentation]",
                f"method '{cfg.pattern.method}' needs a segmentation but the file has no [segmentation] table",
            )
        for seg in sorted(named_wanted):
            if seg not in cfg.segmentations:
                report.error(
                    file,
                    f"[segmentation.{seg}]",
                    f"method '{cfg.pattern.method}' needs segmentation '{seg}', which is not configured",
                )
        if needs_tracks and cfg.tracking is None:
            report.error(
                file,
                "[tracking]",
                f"method '{cfg.pattern.method}' needs tracks but the file has no [tracking] table",
            )
        if (
            cfg.segmentation is not None
            and not needs_default_seg
            and not (
                needs_tracks
                and cfg.tracking is not None
                and cfg.tracking.segmentation == "segmentation"
            )
        ):
            report.warning(
                file,
                "[segmentation]",
                f"configured but the pattern method '{cfg.pattern.method}' does not ask for it; "
                "it will not run",
            )
        for seg in cfg.segmentations:
            tracked = (
                cfg.tracking is not None
                and cfg.tracking.segmentation == seg
                and needs_tracks
            )
            if seg not in named_wanted and not tracked:
                report.warning(
                    file,
                    f"[segmentation.{seg}]",
                    f"configured but nothing asks for it; it will not run",
                )
        if cfg.tracking is not None and not needs_tracks:
            report.warning(
                file,
                "[tracking]",
                f"configured but the pattern method '{cfg.pattern.method}' does not ask for tracks; "
                "it will not run",
            )
        try:
            requirements_by_stem[stem] = method.initialize(experiment)
        except Exception as e:
            report.error(
                file, "[pattern]", f"initialising '{cfg.pattern.method}' failed: {e!r}"
            )

    # 7. MicroManager configuration
    mm = None
    mm_path = None
    if mm_config is not None:
        mm_path = Path(mm_config)
    elif config is not None:
        mm_path = Path(config.config_path)
    if mm_path is not None:
        from .core.real_core import device_interface

        di = device_interface()
        if di is not None:
            report.info(
                "pymmcore",
                "",
                f"device interface {di}: the Micro-Manager adapters at "
                f"{mm_path.parent} must be built for interface {di}",
            )
    if mm_path is not None and mm_path.exists():
        try:
            mm = MMConfig.from_file(mm_path)
            report.info(
                mm_path.name, "", f"MicroManager configuration read: {mm.summary()}"
            )
        except Exception as e:
            report.warning(
                mm_path.name, "", f"could not be read ({e}); presets not checked"
            )
    elif mm_path is not None:
        report.info(
            CONFIG_NAME,
            "config_path",
            f"MicroManager configuration not found at {mm_path}; presets and devices not checked",
        )
    if mm is not None:
        for stem, cfg in configs.items():
            file = f"{stem}.toml"
            for group, preset in sorted(cfg.config_groups_used()):
                if not mm.has_group(group):
                    report.error(
                        file,
                        "",
                        f"config group '{group}' is not in the MicroManager configuration "
                        f"(groups: {', '.join(sorted(mm.groups)) or 'none'})",
                    )
                elif not mm.has_preset(group, preset):
                    report.error(
                        file,
                        "",
                        f"preset '{preset}' is not in config group '{group}' "
                        f"(presets: {', '.join(mm.presets(group)) or 'none'})",
                    )
            for key in sorted(cfg.device_properties_used()):
                device, prop = key.split("-", 1)
                known = mm.has_property(device, prop)
                if known is False:
                    report.error(
                        file,
                        "",
                        f"device '{device}' ('{key}') is not in the MicroManager configuration",
                    )
                elif known is None:
                    report.warning(
                        file,
                        "",
                        f"property '{prop}' of device '{device}' is not listed in the MicroManager "
                        "configuration (it may still exist on the device)",
                    )

    # 8. timing and frame counts
    if schedule is not None and labels and all(s in configs for s in labels.values()):
        from .core.experiments import ExperimentSchedule

        experiments = {
            label: configs[stem].to_experiment(label) for label, stem in labels.items()
        }
        pos_by_label = {p.label: p for p in positions or []}
        requirements = {}
        for label, stem in labels.items():
            method = methods_by_stem.get(stem)
            if method is not None:
                try:
                    requirements[label] = method.initialize(experiments[label])
                except Exception:
                    pass
        try:
            sched = ExperimentSchedule(
                experiments,
                {label: pos_by_label[label] for label in labels},
                **schedule.timing_kwargs(),
            )
            plan = AcquisitionPlan.from_schedule(sched, requirements)
        except Exception as e:
            report.error(SCHEDULE_NAME, "", f"the plan could not be built: {e!r}")
            plan = None
        if plan is not None:
            settle = config.settle_time_seconds if config is not None else 1.0
            over = plan.over_budget(settle_s=settle)
            if over:
                worst_t, worst = max(over, key=lambda item: item[1])
                report.warning(
                    SCHEDULE_NAME,
                    "[timing]",
                    f"{len(over)} of {plan.timepoints} timepoints are estimated to take longer "
                    f"than the {plan.interval_s:g} s interval (worst t={worst_t}: {worst:.1f} s "
                    f"with settle time {settle:g} s); acquisitions will run late",
                )
            counts: dict[str, int] = {}
            for ds in plan.expected_datasets():
                counts[ds.experiment] = counts.get(ds.experiment, 0) + 1
            for label, n in counts.items():
                report.info(label, "", f"{n} frames over {plan.timepoints} timepoints")

    # 9. existing outputs
    for label in labels:
        for suffix in OUTPUT_SUFFIXES:
            out = directory / f"{label}{suffix}"
            if out.exists():
                report.error(
                    out.name,
                    "",
                    "output already exists; move or delete it before running",
                )

    # 10. dry rehearsal
    if dry:
        _dry_rehearsal(
            directory,
            report,
            cfg_path,
            pattern_methods,
            segmentation_methods,
            tracking_methods,
        )
    return report


def _dry_rehearsal(
    directory, report, cfg_path, pattern_methods, segmentation_methods, tracking_methods
):
    """Copy the directory, cut it to two quick timepoints, run it on the virtual microscope."""
    from .run_pyclm import run_pyclm

    if not list(directory.glob("*.tif")) and not (directory / "dry_run.yml").exists():
        report.warning(
            "dry run",
            "",
            "no TIF files in the directory; the virtual microscope needs them",
        )
        return
    tmp = Path(tempfile.mkdtemp(prefix="pyclm-check-"))
    try:
        for item in directory.iterdir():
            if item.suffix in OUTPUT_SUFFIXES or item.name in ("preview", "log.log"):
                continue
            if item.is_dir():
                shutil.copytree(item, tmp / item.name)
            else:
                shutil.copy(item, tmp / item.name)
        schedule = ScheduleConfig.from_file(tmp / SCHEDULE_NAME)
        (tmp / SCHEDULE_NAME).write_text(
            "[timing]\nsteps = 2\ninterval_seconds = 1.0\nsetup_time_seconds = 0.0\n"
            f"time_between_positions = {min(schedule.timing.time_between_positions, 0.1):g}\n"
        )
        run_pyclm(
            tmp,
            None if cfg_path is None else str(cfg_path),
            dry=True,
            pattern_methods=pattern_methods,
            segmentation_methods=segmentation_methods,
            tracking_methods=tracking_methods,
            check=False,
        )
        report.info("dry run", "", "two timepoints ran on the virtual microscope")
    except Exception as e:
        report.error("dry run", "", f"failed: {e!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
