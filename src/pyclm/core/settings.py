"""
Runtime setting changes requested by a pattern method for its own experiment.

A method asks through its :class:`~pyclm.core.patterns.pattern.PatternContext`
(``set_exposure``, ``set_config``, ``set_property``, ``set_position``); the
pattern process ships the collected :class:`SettingChange` records to the
Manager in a :class:`~pyclm.core.messages.SettingsRequestMessage`; the
Manager validates and applies them at the next timepoint boundary
(:meth:`~pyclm.core.manager.Manager.apply_settings`), records each one in
the events table, and stamps the values in force on every later
acquisition event so they appear as columns of the frames table.

The structure of the experiment (channels, cadence, timepoints) never
changes; only the values inside it do. See docs/stage4-control-plane-design.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# the alias a method may use for its experiment's stimulation channel
STIMULATION = "stimulation"
KINDS = ("exposure", "config", "property", "position")
POSITION_KEYS = ("x", "y", "z", "pfs_offset")


@dataclass(frozen=True)
class SettingChange:
    """
    One requested change. ``kind`` is ``"exposure"`` (``key`` =
    ``"exposure_ms"``), ``"config"`` (``key`` = the config group, ``value``
    = the preset), ``"property"`` (``key`` = ``"<device>-<property>"``) or
    ``"position"`` (``key`` in ``x``, ``y``, ``z``, ``pfs_offset``;
    ``channel`` is None).
    """

    kind: str
    channel: str | None
    key: str
    value: Any
    device: str | None = None
    prop: str | None = None

    @property
    def column(self) -> str:
        """The frames-table column that carries this value once changed."""
        return self.key

    def describe(self) -> str:
        where = "position" if self.channel is None else self.channel
        return f"{self.kind} {where}.{self.key} = {self.value!r}"


def property_type(value: Any) -> str:
    """The MicroManager property type string PyCLM uses for a Python value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def check_change(change: SettingChange) -> str | None:
    """Structural validation. Returns a reason to refuse, or None if fine."""
    if change.kind not in KINDS:
        return f"unknown setting kind {change.kind!r}"
    if change.kind == "position":
        if change.key not in POSITION_KEYS:
            return f"unknown position key {change.key!r}"
        try:
            v = float(change.value)
        except (TypeError, ValueError):
            return f"{change.key} must be a number, got {change.value!r}"
        if not math.isfinite(v):
            return f"{change.key} must be finite, got {change.value!r}"
        return None
    if not change.channel:
        return f"{change.kind} needs a channel"
    if change.kind == "exposure":
        try:
            v = float(change.value)
        except (TypeError, ValueError):
            return f"exposure must be a number, got {change.value!r}"
        if not (v > 0 and math.isfinite(v)):
            return f"exposure must be positive, got {change.value!r}"
        return None
    if change.kind == "config":
        if not change.key or not isinstance(change.value, str) or not change.value:
            return "config needs a group and a preset name"
        return None
    if not change.device or not change.prop:
        return "property needs a device and a property name"
    return None


def exposure(channel: str, ms: float) -> SettingChange:
    return SettingChange("exposure", channel, "exposure_ms", float(ms))


def config(channel: str, group: str, preset: str) -> SettingChange:
    return SettingChange("config", channel, str(group), str(preset))


def device_property(channel: str, device: str, prop: str, value: Any) -> SettingChange:
    return SettingChange(
        "property",
        channel,
        f"{device}-{prop}",
        value,
        device=str(device),
        prop=str(prop),
    )


def position(key: str, value: float) -> SettingChange:
    return SettingChange("position", None, key, float(value))
