"""The live viewer (napari, Qt). Installed with the ``gui`` extra."""

from __future__ import annotations

import importlib.util

INSTALL_HINT = (
    'the viewer needs the gui extra: pip install "closed-loop-microscopy[gui]" '
    "(in a clone: uv sync --extra gui)"
)


class GuiUnavailable(RuntimeError):
    """napari or a Qt binding is not installed."""


def require_gui() -> None:
    """Raise :class:`GuiUnavailable`, naming the install line, when the viewer cannot start."""
    missing = [
        name for name in ("napari", "qtpy") if importlib.util.find_spec(name) is None
    ]
    if not any(
        importlib.util.find_spec(binding) is not None
        for binding in ("PySide6", "PyQt6", "PyQt5", "PySide2")
    ):
        missing.append("a Qt binding (PySide6)")
    if missing:
        raise GuiUnavailable(f"{', '.join(missing)} not installed; {INSTALL_HINT}")
