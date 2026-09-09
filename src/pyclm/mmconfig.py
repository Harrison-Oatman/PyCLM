"""
A text reader for MicroManager ``.cfg`` files, so ``pyclm check`` can tell
whether the config groups, presets, devices and properties an experiment
uses exist, without loading the hardware.

Only the lines that matter are read::

    Device,<label>,<library>,<adapter>
    Property,<device>,<property>,<value>
    ConfigGroup,<group>,<preset>,<device>,<property>,<value>

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

    def summary(self) -> str:
        return (
            f"{len(self.devices)} devices, {len(self.groups)} config groups "
            f"({', '.join(sorted(self.groups))})"
        )
