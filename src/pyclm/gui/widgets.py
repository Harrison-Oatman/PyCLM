"""
Plain Qt widgets the viewer shows as a napari dock: a status line built
from ``status.json`` and a minimap of the positions in stage coordinates.
They depend on qtpy only, so any host can show them and none can break the
run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

# a fixed palette, one colour per experiment (stem), cycled
PALETTE = (
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#76b7b2",
    "#edc948",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
)


def status_text(status: dict | None) -> str:
    """One line from a status.json dict."""
    if not status:
        return "no status yet"
    parts = [f"t {status.get('t')}/{status.get('timepoints')}"]
    if status.get("done"):
        parts.append("done")
    elif status.get("paused"):
        parts.append("paused")
    elif status.get("stopping"):
        parts.append("stopping")
    current = status.get("current_experiment")
    if current:
        parts.append(f"at {current}")
    late = [
        f"{name} late {info['lateness_s']:.1f}s"
        for name, info in status.get("experiments", {}).items()
        if info.get("lateness_s")
    ]
    parts += late
    errors = sum(
        info.get("errors", 0) for info in status.get("experiments", {}).values()
    )
    if errors:
        parts.append(f"{errors} acquisition error{'s' if errors != 1 else ''}")
    if status.get("settings_applied"):
        parts.append(f"{status['settings_applied']} settings changed")
    pending = status.get("pending_commands")
    if pending:
        parts.append(f"{pending} command{'s' if pending != 1 else ''} pending")
    return " | ".join(parts)


def read_status(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


class StatusLine(QtWidgets.QLabel):
    """A label showing :func:`status_text`; call :meth:`update_from` every tick."""

    def __init__(self, parent=None):
        super().__init__("no status yet", parent)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.setWordWrap(True)

    def update_from(self, status: dict | None) -> str:
        text = status_text(status)
        self.setText(text)
        return text


@dataclass
class MapPosition:
    label: str
    experiment: str
    x: float
    y: float
    fov_um: tuple[float, float] | None = None  # (height, width) of the field, if known


def positions_from_plan(plan_path: Path) -> list[MapPosition]:
    """Positions of a run from its ``plan.useq.yaml``."""
    import useq

    seq = useq.MDASequence.from_file(str(plan_path))
    out = []
    for pos in seq.stage_positions:
        name = pos.name or ""
        out.append(
            MapPosition(
                name, name.split(".")[0], float(pos.x or 0.0), float(pos.y or 0.0)
            )
        )
    return out


def positions_from_directory(directory: Path) -> list[MapPosition]:
    """Positions from ``plan.useq.yaml`` if a run has been started, else from the position list."""
    directory = Path(directory)
    plan = directory / "plan.useq.yaml"
    if plan.exists():
        try:
            return positions_from_plan(plan)
        except Exception:
            pass
    from ..directories import positions_from_pos, positions_from_xml

    pos_path = directory / "PositionList.pos"
    xml_path = directory / "multipoints.xml"
    if pos_path.exists():
        found = positions_from_pos(str(pos_path))
    elif xml_path.exists():
        found = positions_from_xml(str(xml_path))
    else:
        return []
    return [
        MapPosition(p.label, str(p.label).split(".")[0], float(p.x), float(p.y))
        for p in found
    ]


class Minimap(QtWidgets.QGraphicsView):
    """
    The positions of a run drawn in stage coordinates (µm), coloured by
    experiment, with the field of view drawn when its size is known and the
    position the microscope is at highlighted. Read-only.
    """

    positionClicked = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing
        )
        self.setMinimumSize(160, 160)
        self.positions: list[MapPosition] = []
        self.current: str | None = None
        self.problems: set[str] = set()
        self._colours: dict[str, str] = {}
        self._items: dict[str, QtWidgets.QGraphicsItem] = {}

    # ------------------------------------------------------------ data
    def colour_of(self, experiment: str) -> str:
        if experiment not in self._colours:
            self._colours[experiment] = PALETTE[len(self._colours) % len(PALETTE)]
        return self._colours[experiment]

    def set_positions(self, positions: list[MapPosition]):
        self.positions = list(positions)
        self.redraw()

    def set_current(self, label: str | None):
        if label != self.current:
            self.current = label
            self.redraw()

    def set_status(self, status: dict | None):
        """Highlight the current position and mark experiments with acquisition errors."""
        problems = set()
        if status:
            for name, info in status.get("experiments", {}).items():
                if info.get("errors"):
                    problems.add(name)
        current = status.get("current_experiment") if status else None
        if problems != self.problems or current != self.current:
            self.problems = problems
            self.current = current
            self.redraw()

    # --------------------------------------------------------- drawing
    def redraw(self):
        scene = self._scene
        scene.clear()
        self._items = {}
        if not self.positions:
            scene.addText("no positions")
            return
        xs = [p.x for p in self.positions]
        ys = [p.y for p in self.positions]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        marker = span / 40.0 + 1.0
        for p in self.positions:
            colour = QtGui.QColor(self.colour_of(p.experiment))
            pen = QtGui.QPen(colour)
            pen.setCosmetic(True)
            pen.setWidth(3 if p.label == self.current else 1)
            if p.label in self.problems:
                pen.setColor(QtGui.QColor("#d62728"))
            brush = QtGui.QBrush(
                colour if p.label == self.current else QtCore.Qt.NoBrush
            )
            if p.fov_um:
                h, w = p.fov_um
                item = scene.addRect(p.x - w / 2, p.y - h / 2, w, h, pen, brush)
            else:
                item = scene.addEllipse(
                    p.x - marker, p.y - marker, 2 * marker, 2 * marker, pen, brush
                )
            item.setToolTip(f"{p.label}  x={p.x:.0f} µm  y={p.y:.0f} µm")
            item.setData(0, p.label)
            self._items[p.label] = item
            text = scene.addSimpleText(p.label)
            text.setBrush(QtGui.QBrush(colour))
            text.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            text.setPos(p.x + marker, p.y + marker)
        # a scale bar: 1 mm or 100 µm depending on the span
        bar = 1000.0 if span > 3000 else 100.0
        x0, y0 = min(xs), max(ys) + span * 0.08
        pen = QtGui.QPen(QtGui.QColor("#666666"))
        pen.setCosmetic(True)
        scene.addLine(x0, y0, x0 + bar, y0, pen)
        label = scene.addSimpleText(f"{bar:g} µm")
        label.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        label.setPos(x0, y0)
        self.fit()

    def fit(self):
        rect = self._scene.itemsBoundingRect()
        if rect.isValid():
            self.fitInView(
                rect.adjusted(
                    -rect.width() * 0.1,
                    -rect.height() * 0.1,
                    rect.width() * 0.1,
                    rect.height() * 0.1,
                ),
                QtCore.Qt.KeepAspectRatio,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        label = item.data(0) if item is not None else None
        if label:
            self.positionClicked.emit(str(label))
        super().mousePressEvent(event)


class RunOverview(QtWidgets.QWidget):
    """The status line above the minimap, refreshed from the experiment directory."""

    def __init__(self, directory: Path | None = None, parent=None):
        super().__init__(parent)
        self.directory = None if directory is None else Path(directory)
        self.status = StatusLine()
        self.minimap = Minimap()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.status)
        layout.addWidget(self.minimap, stretch=1)
        self._positions_loaded = False

    def set_directory(self, directory: Path):
        self.directory = Path(directory)
        self._positions_loaded = False
        self.refresh()

    def refresh(self) -> dict | None:
        if self.directory is None:
            return None
        if not self._positions_loaded:
            positions = positions_from_directory(self.directory)
            if positions:
                self.minimap.set_positions(positions)
                self._positions_loaded = True
        status = read_status(self.directory / "status.json")
        self.status.update_from(status)
        self.minimap.set_status(status)
        return status
