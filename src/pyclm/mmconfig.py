"""
A text reader for MicroManager ``.cfg`` files, so ``pyclm check`` can tell
whether the config groups, presets, devices and properties an experiment
uses exist, without loading the hardware.

Only the lines that matter are read::

    Device,<label>,<library>,<adapter>
    Property,<device>,<property>,<value>
    ConfigGroup,<group>,<preset>,<device>,<property>,<value>
    ConfigPixelSize,<preset>,<device>,<property>,<value>
    PixelSize_um,<preset>,<value>
    PixelSizeAffine,<preset>,<a>,<b>,<c>,<d>,<e>,<f>
    Label,<device>,<state>,<label>

A property that never appears in the file may still exist on the device,
so :meth:`MMConfig.has_property` returns None for "unknown" rather than
False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CORE_DEVICE = "Core"


@dataclass
class MMConfig:
    path: Path | None = None
    devices: set[str] = field(default_factory=lambda: {CORE_DEVICE})
    properties: dict[str, set[str]] = field(default_factory=dict)
    # group -> preset -> [(device, property, value), ...]
    groups: dict[str, dict[str, list[tuple[str, str, str]]]] = field(
        default_factory=dict
    )
    # pixel-size preset -> camera-to-stage affine [a, b, c, d, e, f]
    pixel_affines: dict[str, list[float]] = field(default_factory=dict)
    # state device -> state -> label (``Label,<device>,<state>,<label>``)
    labels: dict[str, dict[str, str]] = field(default_factory=dict)
    # pixel-size preset -> (um per pixel, [(device, property, value), ...])
    pixel_sizes: dict[str, tuple[float | None, list[tuple[str, str, str]]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_file(cls, path: str | Path) -> MMConfig:
        path = Path(path)
        cfg = cls(path=path)
        with open(path, encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                kind = parts[0]
                if kind == "Device" and len(parts) >= 2:
                    cfg.devices.add(parts[1])
                elif kind == "Property" and len(parts) >= 3:
                    cfg.devices.add(parts[1])
                    cfg.properties.setdefault(parts[1], set()).add(parts[2])
                elif kind == "ConfigPixelSize" and len(parts) >= 5:
                    _um, conds = cfg.pixel_sizes.setdefault(parts[1], (None, []))
                    conds.append((parts[2], parts[3], ",".join(parts[4:])))
                elif kind == "PixelSize_um" and len(parts) >= 3:
                    _um, conds = cfg.pixel_sizes.setdefault(parts[1], (None, []))
                    try:
                        cfg.pixel_sizes[parts[1]] = (float(parts[2]), conds)
                    except ValueError:
                        pass
                elif kind == "PixelSizeAffine" and len(parts) >= 8:
                    try:
                        cfg.pixel_affines[parts[1]] = [float(v) for v in parts[2:8]]
                    except ValueError:
                        pass
                elif kind == "Label" and len(parts) >= 4:
                    cfg.labels.setdefault(parts[1], {})[parts[2]] = ",".join(parts[3:])
                elif kind == "ConfigGroup" and len(parts) >= 3:
                    group = cfg.groups.setdefault(parts[1], {})
                    preset = group.setdefault(parts[2], [])
                    if len(parts) >= 6:
                        preset.append((parts[3], parts[4], ",".join(parts[5:])))
                        cfg.devices.add(parts[3])
                        cfg.properties.setdefault(parts[3], set()).add(parts[4])
        return cfg

    def has_group(self, group: str) -> bool:
        return group in self.groups

    def has_preset(self, group: str, preset: str) -> bool:
        return preset in self.groups.get(group, {})

    def presets(self, group: str) -> list[str]:
        return sorted(self.groups.get(group, {}))

    def has_device(self, label: str) -> bool:
        return label in self.devices

    def has_property(self, device: str, prop: str) -> bool | None:
        """True / False when the device is known, None when the property is simply not listed."""
        if device not in self.devices:
            return False
        return True if prop in self.properties.get(device, set()) else None

    def device_state(
        self, presets: list[tuple[str, str]]
    ) -> dict[tuple[str, str], str]:
        """The (device, property) -> value the given (group, preset) pairs set."""
        state: dict[tuple[str, str], str] = {}
        for group, preset in presets:
            for device, prop, value in self.groups.get(group, {}).get(preset, []):
                state[(device, prop)] = value
        # a state device's Label and State are two names for one position:
        # a preset that sets one also sets the other
        for (device, prop), value in list(state.items()):
            labels = self.labels.get(device, {})
            if prop == "Label":
                for st, label in labels.items():
                    if label.strip() == str(value).strip():
                        state.setdefault((device, "State"), st)
            elif prop == "State" and str(value).strip() in labels:
                state.setdefault((device, "Label"), labels[str(value).strip()])
        return state

    def pixel_size_candidates(self, presets: list[tuple[str, str]]) -> list[dict]:
        """
        Every pixel-size preset against the device state the given (group,
        preset) pairs produce: ``{"name", "um", "unmet", "unknown"}`` where
        ``unmet`` lists conditions the state contradicts and ``unknown``
        conditions on properties no given preset sets (MicroManager would
        use the device's current value for those). Exact matches first.
        """
        state = self.device_state(presets)
        out = []
        for name, (um, conds) in self.pixel_sizes.items():
            unmet, unknown = [], []
            for device, prop, value in conds:
                have = state.get((device, prop))
                if have is None:
                    unknown.append((device, prop, value))
                elif str(have).strip() != str(value).strip():
                    unmet.append((device, prop, value, have))
            out.append({"name": name, "um": um, "unmet": unmet, "unknown": unknown})
        out.sort(key=lambda c: (len(c["unmet"]), len(c["unknown"]), c["name"]))
        return out

    def pixel_size_for(
        self, presets: list[tuple[str, str]]
    ) -> tuple[str, float | None] | None:
        """
        The pixel-size preset whose conditions are all met by the device
        state the given (group, preset) pairs produce, as (name, um), or
        None. MicroManager resolves the pixel size the same way: every
        condition of a preset must match the current device properties.
        """
        for c in self.pixel_size_candidates(presets):
            if not c["unmet"] and not c["unknown"]:
                return c["name"], c["um"]
        return None

    def affine_form(self, preset: str) -> str | None:
        """
        How a pixel-size preset's camera-to-stage affine maps the camera axes,
        e.g. ``"x -> +x, y -> -y"``; None when the preset has no affine or it
        is all zeros (MicroManager then assumes the identity).
        """
        aff = self.pixel_affines.get(preset)
        if not aff or not any(aff[:2] + aff[3:5]):
            return None
        a, b, _c, d, e, _f = aff
        out = []
        # column 0 is where camera x goes, column 1 where camera y goes
        for axis, (sx, sy) in (("x", (a, d)), ("y", (b, e))):
            to = "x" if abs(sx) >= abs(sy) else "y"
            sign = "+" if (sx if to == "x" else sy) >= 0 else "-"
            out.append(f"{axis} -> {sign}{to}")
        return ", ".join(out)

    def summary(self) -> str:
        return (
            f"{len(self.devices)} devices, {len(self.groups)} config groups "
            f"({', '.join(sorted(self.groups))})"
        )
