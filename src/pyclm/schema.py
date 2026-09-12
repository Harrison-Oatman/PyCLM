"""
The configuration schema: what the experiment TOML, ``schedule.toml`` and
``pyclm_config.toml`` may contain, with defaults and limits, and plain
messages when they do not.

The models validate and then build the same objects the rest of PyCLM has
always used (:class:`~pyclm.core.experiments.Experiment`, the timing dict
of ``ExperimentSchedule``), so nothing downstream changes. Unknown keys are
errors everywhere except in the three method tables (``[segmentation]``,
``[tracking]``, ``[pattern]``), whose extra keys are the method's keyword
arguments; ``pyclm check`` compares those with the method's signature.

See docs/stage5-schema-setup-design.md §2.
"""

from __future__ import annotations

import difflib
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from toml import load as _load_toml

from .core.experiments import (
    ConfigGroup,
    DeviceProperty,
    Experiment,
    ImagingConfig,
    PatternConfig,
    SegmentationConfig,
    TrackingConfig,
    check_type,
)
from .core.kinds import DEFAULT_SEGMENTATION

FORMAT_VERSION = 1

Scalar = bool | int | float | str


class ConfigError(ValueError):
    """A configuration file has problems; ``problems`` lists them, one line each."""

    def __init__(self, file: str | Path | None, problems: list[str]):
        self.file = None if file is None else str(file)
        self.problems = list(problems)
        prefix = "" if self.file is None else f"{Path(self.file).name}: "
        super().__init__("\n".join(prefix + p for p in self.problems))


# ------------------------------------------------------------------ pieces
class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_version(v: int) -> int:
    if v > FORMAT_VERSION:
        raise ValueError(
            f"format_version {v} is newer than this PyCLM understands ({FORMAT_VERSION})"
        )
    if v < 1:
        raise ValueError("format_version must be at least 1")
    return v


def _check_device_properties(table: dict) -> dict:
    for key in table:
        if key.count("-") != 1 or not all(key.split("-")):
            raise ValueError(
                f'device property key {key!r} must be written "Device-Property"'
            )
    return table


class ImagingDefaults(Strict):
    """Defaults for every imaging channel."""

    exposure: float = Field(10.0, gt=0, description="exposure in ms")
    every_t: int = Field(
        1, ge=1, description="acquire every N timepoints, counted from t_delay"
    )
    save: bool = Field(True, description="record the frames")
    binning: int = Field(
        1, ge=1, description="camera binning, the same for every channel"
    )
    config_groups: dict[str, str] = Field(
        default_factory=dict, description="group = preset, applied to imaging channels"
    )
    device_properties: dict[str, Scalar] = Field(
        default_factory=dict,
        description='"Device-Property" = value, applied to imaging channels',
    )

    @model_validator(mode="after")
    def _props(self):
        _check_device_properties(self.device_properties)
        return self


class ChannelOverride(Strict):
    """Per-preset overrides of the imaging defaults (``[channels.<preset>]``)."""

    exposure: float | None = Field(None, gt=0, description="exposure in ms")
    every_t: int | None = Field(None, ge=1, description="acquire every N timepoints")
    config_groups: dict[str, str] = Field(
        default_factory=dict, description="group = preset"
    )
    device_properties: dict[str, Scalar] = Field(
        default_factory=dict, description='"Device-Property" = value'
    )

    @model_validator(mode="before")
    @classmethod
    def _no_binning(cls, data):
        if isinstance(data, dict) and "binning" in data:
            raise ValueError(
                "binning cannot differ between channels; set it once under [imaging]"
            )
        return data

    @model_validator(mode="after")
    def _props(self):
        _check_device_properties(self.device_properties)
        return self


class Channels(BaseModel):
    """
    Which presets of which config group are the imaging channels. May be
    empty (or the table absent) for an experiment that only stimulates.
    """

    model_config = ConfigDict(extra="forbid")

    group: str = Field(
        "Channel", description="the MicroManager config group that switches channels"
    )
    presets: list[str] = Field(
        default_factory=list,
        description="the presets to image, in acquisition order; empty for a "
        "stimulation-only experiment",
    )
    overrides: dict[str, ChannelOverride] = Field(
        default_factory=dict,
        description="[channels.<preset>] tables: exposure, every_t, config_groups, device_properties",
    )

    @model_validator(mode="before")
    @classmethod
    def _collect_overrides(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        overrides = dict(data.pop("overrides", {}) or {})
        for key in list(data):
            if key not in ("group", "presets") and isinstance(data[key], dict):
                overrides[key] = data.pop(key)
        data["overrides"] = overrides
        return data

    @model_validator(mode="after")
    def _consistent(self):
        if len(set(self.presets)) != len(self.presets):
            raise ValueError(f"presets contains duplicates: {self.presets}")
        for name in self.overrides:
            if name not in self.presets:
                raise ValueError(
                    f"[channels.{name}] names a preset that is not in presets {self.presets}"
                )
        return self


class Stimulation(Strict):
    """The light-delivery event (the DMD pattern is applied here)."""

    exposure: float = Field(
        ..., ge=0, description="exposure in ms; 0 means no stimulation"
    )
    every_t: int = Field(1, ge=1, description="stimulate every N timepoints")
    save: bool = Field(
        True, description="record the camera frame taken during stimulation"
    )
    binning: int | None = Field(None, ge=1, description="default: the imaging binning")
    config_groups: dict[str, str] = Field(
        default_factory=dict, description="group = preset"
    )
    device_properties: dict[str, Scalar] = Field(
        default_factory=dict, description='"Device-Property" = value'
    )

    @model_validator(mode="after")
    def _props(self):
        _check_device_properties(self.device_properties)
        return self


# keys that belong at the top of the experiment file; a TOML key written after
# a table header lands inside that table, which is where beginners lose them
TOP_LEVEL_KEYS = ("t_delay", "t_stop")


class MethodTable(BaseModel):
    """A method and its keyword arguments; extra keys are passed to the method."""

    model_config = ConfigDict(extra="allow")

    method: str = Field(..., description="a built-in or registered method name")
    save: bool = Field(True, description="record the method's output")

    @model_validator(mode="before")
    @classmethod
    def _save_alias(cls, data):
        if isinstance(data, dict) and "save_output" in data:
            data = dict(data)
            if "save" in data:
                raise ValueError("give either save or save_output, not both")
            data["save"] = data.pop("save_output")
        if isinstance(data, dict):
            misplaced = [k for k in TOP_LEVEL_KEYS if k in data]
            if misplaced:
                raise ValueError(
                    f"{', '.join(misplaced)} must be written at the top of the file, "
                    "before the first table (a key written after a table header "
                    "belongs to that table)"
                )
        return data

    @property
    def kwargs(self) -> dict[str, Any]:
        """The keys passed to the method's constructor."""
        return dict(self.model_extra or {})


class SegmentationTable(MethodTable):
    """``[segmentation]`` or a named ``[segmentation.<name>]``."""


class TrackingTable(MethodTable):
    """``[tracking]``."""

    segmentation: str = Field(
        DEFAULT_SEGMENTATION,
        description="the segmentation table whose objects are tracked",
    )


class PatternTable(MethodTable):
    """``[pattern]``."""

    every_t: int = Field(
        1, ge=1, description="regenerate the pattern every N timepoints"
    )


class ExperimentConfig(Strict):
    """One experiment TOML."""

    format_version: int = Field(
        FORMAT_VERSION, description="schema version of this file"
    )
    config_groups: dict[str, str] = Field(
        default_factory=dict, description="group = preset, applied to every acquisition"
    )
    device_properties: dict[str, Scalar] = Field(
        default_factory=dict,
        description='"Device-Property" = value, applied to every acquisition',
    )
    imaging: ImagingDefaults = Field(default_factory=ImagingDefaults)
    channels: Channels = Field(
        default_factory=Channels,
        description="the imaging channels; omit for a stimulation-only experiment",
    )
    stimulation: Stimulation
    segmentation: SegmentationTable | None = Field(
        None, description="the default segmentation"
    )
    segmentations: dict[str, SegmentationTable] = Field(
        default_factory=dict, description="named [segmentation.<name>] tables"
    )
    tracking: TrackingTable | None = None
    pattern: PatternTable
    t_delay: int = Field(
        0, ge=0, description="timepoints to wait before this experiment starts"
    )
    t_stop: int = Field(
        0,
        ge=0,
        description="stop after this many of its own timepoints (0 = run to the end)",
    )

    @model_validator(mode="before")
    @classmethod
    def _split_named_segmentations(cls, data):
        """``[segmentation.<name>]`` sub-tables become ``segmentations``."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        seg = data.get("segmentation")
        if isinstance(seg, dict):
            seg = dict(seg)
            named = dict(data.get("segmentations") or {})
            for key in list(seg):
                if isinstance(seg[key], dict):
                    named[key] = seg.pop(key)
            data["segmentations"] = named
            data["segmentation"] = seg if seg else None
        return data

    @model_validator(mode="after")
    def _cross(self):
        _check_device_properties(self.device_properties)
        _check_version(self.format_version)
        if DEFAULT_SEGMENTATION in self.segmentations:
            raise ValueError(
                f"[segmentation.{DEFAULT_SEGMENTATION}] is reserved for the default [segmentation] table"
            )
        if self.tracking is not None:
            name = self.tracking.segmentation
            have = self.segmentation_names
            if name not in have:
                raise ValueError(
                    f"[tracking] segmentation = {name!r} but the experiment configures "
                    + (f"only {have}" if have else "no segmentation")
                )
        return self

    # ----------------------------------------------------------- queries
    @property
    def segmentation_names(self) -> list[str]:
        names = [DEFAULT_SEGMENTATION] if self.segmentation is not None else []
        return names + list(self.segmentations)

    @property
    def channel_names(self) -> list[str]:
        return list(self.channels.presets)

    def config_groups_used(self) -> set[tuple[str, str]]:
        """Every (group, preset) the experiment sets, including the channel group."""
        used = set(self.config_groups.items()) | set(self.imaging.config_groups.items())
        used |= {(self.channels.group, p) for p in self.channels.presets}
        for ov in self.channels.overrides.values():
            used |= set(ov.config_groups.items())
        used |= set(self.stimulation.config_groups.items())
        return used

    def device_properties_used(self) -> set[str]:
        keys = set(self.device_properties) | set(self.imaging.device_properties)
        for ov in self.channels.overrides.values():
            keys |= set(ov.device_properties)
        keys |= set(self.stimulation.device_properties)
        return keys

    # ------------------------------------------------------------- build
    def to_experiment(self, name: str) -> Experiment:
        """The runtime ``Experiment`` this file describes, for position ``name``."""
        base = ImagingConfig(
            experiment_name=name,
            config_groups=_groups(self.config_groups),
            device_properties=_props(self.device_properties),
        )

        imaging = deepcopy(base)
        imaging.set_id()
        imaging.update_config_groups(_groups(self.imaging.config_groups))
        imaging.update_device_properties(_props(self.imaging.device_properties))
        imaging.exposure = self.imaging.exposure
        imaging.every_t = self.imaging.every_t
        imaging.save = self.imaging.save
        imaging.binning = self.imaging.binning

        channels = {}
        for preset in self.channels.presets:
            cfg = deepcopy(imaging)
            cfg.set_id()
            cfg.update_config_groups([ConfigGroup(self.channels.group, preset)])
            ov = self.channels.overrides.get(preset)
            if ov is not None:
                if ov.exposure is not None:
                    cfg.exposure = ov.exposure
                if ov.every_t is not None:
                    cfg.every_t = ov.every_t
                cfg.update_config_groups(_groups(ov.config_groups))
                cfg.update_device_properties(_props(ov.device_properties))
            channels[preset] = cfg

        stim = deepcopy(base)
        stim.set_id()
        stim.exposure = self.stimulation.exposure
        stim.every_t = self.stimulation.every_t
        stim.save = self.stimulation.save
        stim.binning = (
            self.imaging.binning
            if self.stimulation.binning is None
            else self.stimulation.binning
        )
        stim.update_config_groups(_groups(self.stimulation.config_groups))
        stim.update_device_properties(_props(self.stimulation.device_properties))

        segmentation = (
            SegmentationConfig("none")
            if self.segmentation is None
            else SegmentationConfig(
                self.segmentation.method,
                save_output=self.segmentation.save,
                **self.segmentation.kwargs,
            )
        )
        segmentations = {
            n: SegmentationConfig(t.method, save_output=t.save, **t.kwargs)
            for n, t in self.segmentations.items()
        }
        tracking = (
            TrackingConfig("none")
            if self.tracking is None
            else TrackingConfig(
                self.tracking.method,
                save_output=self.tracking.save,
                segmentation=self.tracking.segmentation,
                **self.tracking.kwargs,
            )
        )
        pattern = PatternConfig(
            self.pattern.method,
            save_output=self.pattern.save,
            every_t=self.pattern.every_t,
            **self.pattern.kwargs,
        )
        return Experiment(
            experiment_name=name,
            imaging_configs=channels,
            stimulation_config=stim,
            segmentation=segmentation,
            pattern=pattern,
            t_delay=self.t_delay,
            t_stop=self.t_stop,
            tracking=tracking,
            segmentations=segmentations,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        return load_model(cls, path)


class Timing(Strict):
    """``[timing]`` of ``schedule.toml``."""

    steps: int = Field(..., ge=1, description="number of timepoints")
    interval_seconds: float = Field(..., gt=0, description="seconds between timepoints")
    setup_time_seconds: float = Field(
        2.0,
        ge=0,
        description="seconds before each timepoint at which its events are prepared",
    )
    time_between_positions: float = Field(
        2.0, ge=0, description="seconds between successive positions within a timepoint"
    )


class ScheduleConfig(Strict):
    """``schedule.toml``."""

    format_version: int = Field(
        FORMAT_VERSION, description="schema version of this file"
    )
    timing: Timing

    @model_validator(mode="after")
    def _version(self):
        _check_version(self.format_version)
        return self

    def timing_kwargs(self) -> dict:
        """What ``ExperimentSchedule`` takes."""
        return {
            "t_count": self.timing.steps,
            "t_interval": self.timing.interval_seconds,
            "t_setup": self.timing.setup_time_seconds,
            "t_between": self.timing.time_between_positions,
        }

    @classmethod
    def from_file(cls, path: str | Path) -> ScheduleConfig:
        return load_model(cls, path)


class OutputConfig(Strict):
    """``[output]`` of ``pyclm_config.toml``."""

    format: Literal["ome-zarr", "hdf5"] = Field(
        "ome-zarr", description="storage format"
    )
    pattern_policy: Literal["on_change", "imaging", "none"] = Field(
        "on_change",
        description="which patterns to store (ome-zarr only): every distinct one "
        "(on_change), only the one in force at timepoints where an imaging frame "
        "is saved (imaging), or none",
    )
    export_imagej: bool = Field(
        True, description="write ImageJ hyperstacks when the run ends"
    )


class PyclmConfig(Strict):
    """``pyclm_config.toml``: the hardware and output configuration."""

    format_version: int = Field(
        FORMAT_VERSION, description="schema version of this file"
    )
    config_path: str = Field(..., description="path to the MicroManager .cfg file")
    affine_transform: list[list[float]] = Field(
        ..., description="2 x 3 camera-to-SLM affine matrix [[a, b, tx], [c, d, ty]]"
    )
    slm_shape_h: int = Field(..., gt=0, description="SLM (DMD) height in pixels")
    slm_shape_w: int = Field(..., gt=0, description="SLM (DMD) width in pixels")
    focus_device: str = Field(
        "ZDrive", description="MicroManager focus (Z) device selected at start-up"
    )
    settle_time_seconds: float = Field(
        1.0,
        ge=0,
        description="seconds to wait after the hardware reports ready, before each snap",
    )
    camera_roi: list[int] | None = Field(
        None,
        description="camera ROI [x, y, width, height] in unbinned pixels, set when a run "
        "starts (for example the region the DMD covers; pyclm check prints it); "
        "omit to leave the camera as it is",
    )
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _shape(self):
        _check_version(self.format_version)
        at = self.affine_transform
        if len(at) != 2 or any(len(row) != 3 for row in at):
            raise ValueError(
                "affine_transform must be a 2 x 3 matrix [[a, b, tx], [c, d, ty]]"
            )
        roi = self.camera_roi
        if roi is not None:
            if len(roi) != 4 or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0:
                raise ValueError(
                    "camera_roi must be [x, y, width, height] with x, y >= 0 and "
                    "width, height > 0 (unbinned pixels)"
                )
        return self

    def dmd_footprint(self) -> tuple[int, int, int, int]:
        """
        The camera region the DMD can reach, ``[x, y, width, height]`` in
        unbinned camera pixels: the DMD rectangle mapped through the inverse
        of the camera-to-SLM affine.
        """
        import cv2

        h, w = self.slm_shape
        inv = cv2.invertAffineTransform(self.affine)
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        cam = corners @ inv[:, :2].T + inv[:, 2]
        x0, y0 = np.floor(cam.min(axis=0))
        x1, y1 = np.ceil(cam.max(axis=0))
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    @property
    def affine(self) -> np.ndarray:
        return np.array(self.affine_transform, dtype=np.float32)

    @property
    def slm_shape(self) -> tuple[int, int]:
        return (self.slm_shape_h, self.slm_shape_w)

    @classmethod
    def from_file(cls, path: str | Path) -> PyclmConfig:
        return load_model(cls, path)


# ------------------------------------------------------------- loading
def _groups(table: dict[str, str]) -> list[ConfigGroup]:
    return [ConfigGroup(g, p) for g, p in table.items()]


def _props(table: dict[str, Scalar]) -> list[DeviceProperty]:
    out = []
    for key, value in table.items():
        device, prop = key.split("-", 1)
        out.append(DeviceProperty(device, prop, value, check_type(value)))
    return out


def read_toml(path: str | Path) -> dict:
    path = Path(path)
    try:
        with open(path) as f:
            return _load_toml(f)
    except FileNotFoundError:
        raise ConfigError(path, ["file not found"]) from None
    except Exception as e:  # toml.TomlDecodeError and friends
        raise ConfigError(path, [f"not valid TOML: {e}"]) from None


def validate_model(model: type[BaseModel], data: dict, file: str | Path | None = None):
    """``model.model_validate(data)``, raising :class:`ConfigError` with readable problems."""
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(file, describe_errors(e, model)) from None


def load_model(model: type[BaseModel], path: str | Path):
    return validate_model(model, read_toml(path), path)


# ------------------------------------------------------- error messages
def _table_name(loc: tuple) -> str:
    """The table at ``loc`` as the file writes it: ``[channels.638]``, ``[segmentation.nuclei]``."""
    parts = [str(p) for p in loc if str(p) != "overrides"]
    if parts and parts[0] == "segmentations":
        parts[0] = "segmentation"
    if not parts:
        return "the file"
    return "[" + ".".join(parts) + "]"


def _fields_at(
    model: type[BaseModel] | None, loc: tuple
) -> tuple[list[str], str | None]:
    """The allowed keys of the table at ``loc`` and the table's display name."""
    if model is None:
        return [], None
    current: Any = model
    for part in loc:
        if isinstance(current, type) and issubclass(current, BaseModel):
            field = current.model_fields.get(str(part))
            if field is None:
                return [], None
            current = field.annotation
        else:
            # dict[str, Model]: the part is a key, the value type is the model
            args = getattr(current, "__args__", ())
            current = args[1] if len(args) == 2 else None
        # unwrap Optional[Model]
        if current is not None and getattr(current, "__origin__", None) is not None:
            args = [a for a in current.__args__ if a is not type(None)]
            if len(args) == 1:
                current = args[0]
        if current is None:
            return [], None
    if isinstance(current, type) and issubclass(current, BaseModel):
        names = [n for n in current.model_fields if n != "overrides"]
        return sorted(names), current.__name__
    return [], None


def describe_errors(
    error: ValidationError, model: type[BaseModel] | None = None
) -> list[str]:
    """One readable line per problem, in the vocabulary of the TOML file."""
    lines = []
    for err in error.errors():
        loc = tuple(err.get("loc", ()))
        kind = err.get("type", "")
        msg = err.get("msg", "")
        given = err.get("input")
        # display names: the last element is the key, the rest the table
        if kind == "extra_forbidden":
            table, key = loc[:-1], str(loc[-1])
            allowed, _ = _fields_at(model, table)
            hint = ""
            close = difflib.get_close_matches(key, allowed, n=1)
            if close:
                hint = f" (did you mean {close[0]!r}?)"
            where = _table_name(table)
            lines.append(
                f"unknown key {key!r} in {where}{hint}"
                + (f"; allowed: {', '.join(allowed)}" if allowed else "")
            )
        elif kind == "missing":
            table, key = loc[:-1], str(loc[-1])
            lines.append(f"{_table_name(table)} is missing the required key {key!r}")
        elif kind == "value_error":
            # our own ValueErrors: strip pydantic's prefix
            text = msg.removeprefix("Value error, ")
            where = _table_name(loc)
            lines.append(text if not loc else f"{where}: {text}")
        else:
            table, key = loc[:-1], (str(loc[-1]) if loc else "")
            text = msg
            if kind in (
                "greater_than",
                "greater_than_equal",
                "less_than",
                "less_than_equal",
            ):
                text = text.replace("Input should be", "must be")
            elif kind.endswith("_type") or kind.endswith("_parsing"):
                text = "must be " + msg.removeprefix(
                    "Input should be a valid "
                ).removeprefix("Input should be ")
            shown = (
                f" (got {given!r})"
                if given is not None and not isinstance(given, dict)
                else ""
            )
            lines.append(
                f"{_table_name(table)} {key}: {text}{shown}".replace("  ", " ")
            )
    return lines


# ------------------------------------------------------------- helpers
def experiment_from_file(path: str | Path, name: str) -> Experiment:
    return ExperimentConfig.from_file(path).to_experiment(name)


def schedule_timing_from_file(path: str | Path) -> dict:
    return ScheduleConfig.from_file(path).timing_kwargs()


def pyclm_config_from_file(path: str | Path) -> PyclmConfig:
    return PyclmConfig.from_file(path)


MODELS = {
    "ExperimentConfig": ExperimentConfig,
    "ImagingDefaults": ImagingDefaults,
    "Channels": Channels,
    "ChannelOverride": ChannelOverride,
    "Stimulation": Stimulation,
    "SegmentationTable": SegmentationTable,
    "TrackingTable": TrackingTable,
    "PatternTable": PatternTable,
    "ScheduleConfig": ScheduleConfig,
    "Timing": Timing,
    "PyclmConfig": PyclmConfig,
    "OutputConfig": OutputConfig,
}


def field_table(model: type[BaseModel]) -> list[dict]:
    """Rows describing a model's fields (for the generated documentation)."""
    rows = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        type_name = _type_name(annotation)
        if field.is_required():
            default = "required"
        elif field.default_factory is not None:
            default = "{}" if type_name.startswith("table") else "defaults"
        else:
            default = repr(field.default)
        limits = []
        for meta in field.metadata:
            for attr, sym in (
                ("gt", ">"),
                ("ge", "≥"),
                ("lt", "<"),
                ("le", "≤"),
                ("min_length", "≥ length"),
            ):
                v = getattr(meta, attr, None)
                if v is not None:
                    limits.append(f"{sym} {v}")
        rows.append(
            {
                "key": name,
                "type": type_name + (f" ({', '.join(limits)})" if limits else ""),
                "default": default,
                "description": field.description or "",
            }
        )
    return rows


def _type_name(annotation) -> str:
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    if origin is dict:
        return "table"
    if origin is list:
        return f"list of {_type_name(args[0])}" if args else "list"
    if origin is Literal:
        return " | ".join(repr(a) for a in args)
    if args and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        return _type_name(inner[0]) if len(inner) == 1 else "value"
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return "table"
        return annotation.__name__
    return str(annotation).replace("typing.", "")
