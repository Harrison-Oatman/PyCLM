"""
Live napari viewer for a running (or finished) PyCLM experiment directory.

Reads every output through ``pyclm.io`` so it works for both the HDF5 (format 1)
and OME-Zarr (format 2) layouts.

Positions are an axis, not layers: every channel of a cadence group is one
layer whose data is stacked over (position, t, y, x), so the layer list has
one entry per channel however many positions are imaged, the position slider
(or the positions list in the dock, the minimap, the ``[`` / ``]`` keys)
switches between positions, and contrast limits are one setting per channel
that applies to every position. Groups of different cadence get their own
layers with the time scale set to the group's ``every_t``, so they line up
on the plan's time axis without blank frames. A cyan overlay shows the DMD
pattern in force at each frame when the affine transform is known.

Contrast is normalised automatically as frames arrive until you touch a
layer's limits; then they stay where you put them ("Auto-contrast" in the
dock turns it back on, "Normalise now" resets every channel once).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("NAPARI_DISABLE_PLUGIN_AUTOLOAD", "1")

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import napari
import numpy as np
from qtpy import QtCore, QtWidgets

from pyclm import io as pyclm_io
from pyclm.io.export import pattern_to_camera

from .widgets import RunOverview

logger = logging.getLogger(__name__)

AXIS_LABELS = ("position", "t", "y", "x")


def _load_layers_file(path: Path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    layers = []
    try:
        data = json.loads(text)
        entries = data["all_layers"] if isinstance(data, dict) else data
    except json.JSONDecodeError:
        entries = [line for line in text.splitlines() if line.strip()]
    for entry in entries:
        path_str, layer = entry.rsplit(":", 1)
        layers.append((path_str.strip().strip('"'), layer.strip()))
    return layers


def _parse_src(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            'Each --src must be in the form "path:group/channel"'
        )
    path, layer = s.rsplit(":", 1)
    return path.strip().strip('"'), layer.strip()


def _resolve_layer(exp: pyclm_io.ExperimentData, layer: str) -> tuple[str, str] | None:
    """Map a layer key ("imaging/545", "channel_545", "stim_aq") to (group, channel)."""
    if "/" in layer:
        group, channel = layer.split("/", 1)
        if group in exp.groups and channel in exp.groups[group].channels:
            return group, channel
        return None
    for name, g in exp.groups.items():
        if name == layer or (
            layer.startswith("channel_")
            and name == layer.replace("channel_", "imaging_", 1)
        ):
            return name, g.channels[0]
        if layer in g.channels:
            return name, layer
    return None


# ================================================================ stacks
@dataclass(eq=False)  # hashed by identity: stacks live in sets
class ChannelStack:
    """
    One napari layer holding one channel of one cadence group for every
    position: data (position, t, y, x). ``every_t`` / ``t_delay`` are the
    cadence the layer's time axis is scaled to.
    """

    key: str
    every_t: int
    t_delay: int
    layer: napari.layers.Image
    loaded: set[tuple[int, int]] = field(default_factory=set)
    auto_contrast: bool = True
    _resetting: bool = False

    @property
    def data(self) -> np.ndarray:
        return self.layer.data

    def put(self, p: int, i: int, frame: np.ndarray) -> None:
        """Store ``frame`` at (position p, local timepoint i), growing the stack as needed."""
        data = self.layer.data
        n_p, n_t = data.shape[0], data.shape[1]
        h = max(data.shape[2], frame.shape[0])
        w = max(data.shape[3], frame.shape[1])
        grow = p >= n_p or i >= n_t or (h, w) != data.shape[2:]
        if grow:
            new = np.zeros((max(n_p, p + 1), max(n_t, i + 1), h, w), data.dtype)
            new[:n_p, :n_t, : data.shape[2], : data.shape[3]] = data
            data = new
        data[p, i, : frame.shape[0], : frame.shape[1]] = frame
        if grow:
            self.layer.data = data
        else:
            self.layer.refresh()
        self.loaded.add((p, i))

    def reset_contrast(self) -> None:
        self._resetting = True
        try:
            self.layer.reset_contrast_limits()
        finally:
            self._resetting = False

    def _on_contrast_changed(self, event=None) -> None:
        # the user moved the slider (our own resets are flagged): keep it
        if not self._resetting:
            self.auto_contrast = False


class Stacks:
    """The shared layers of a viewer, keyed by cadence group / channel."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.channels: dict[str, ChannelStack] = {}
        self.patterns: dict[str, ChannelStack] = {}
        self._cadence: dict[str, tuple[int, int]] = {}

    def group_key(self, group: str, every_t: int, t_delay: int) -> str:
        """
        The layer name for a cadence group: the group's name, or, if another
        experiment uses the same name at a different cadence, name@every_t.
        """
        key = group
        seen = self._cadence.setdefault(key, (every_t, t_delay))
        if seen != (every_t, t_delay):
            key = f"{group}@{every_t}"
            if t_delay:
                key += f"+{t_delay}"
            self._cadence.setdefault(key, (every_t, t_delay))
        return key

    def _new_layer(self, name, every_t, t_delay, shape, **kwargs):
        data = np.zeros((1, 1, *shape), dtype=np.uint16)
        return self.viewer.add_image(
            data,
            name=name,
            scale=(1, every_t, 1, 1),
            translate=(0, t_delay, 0, 0),
            **kwargs,
        )

    def channel(self, gkey: str, channel: str, every_t, t_delay, shape) -> ChannelStack:
        key = f"{gkey}/{channel}"
        stack = self.channels.get(key)
        if stack is None:
            layer = self._new_layer(key, every_t, t_delay, shape)
            stack = ChannelStack(key, every_t, t_delay, layer)
            layer.events.contrast_limits.connect(stack._on_contrast_changed)
            self.channels[key] = stack
        return stack

    def pattern(self, gkey: str, every_t, t_delay, shape) -> ChannelStack:
        key = f"{gkey}/pattern"
        stack = self.patterns.get(key)
        if stack is None:
            layer = self._new_layer(
                key,
                every_t,
                t_delay,
                shape,
                colormap="cyan",
                opacity=0.25,
                contrast_limits=(0, 255),
                blending="additive",
            )
            stack = ChannelStack(key, every_t, t_delay, layer, auto_contrast=False)
            self.patterns[key] = stack
        return stack

    def all(self) -> list[ChannelStack]:
        return [*self.channels.values(), *self.patterns.values()]


# ============================================================ experiments
class LiveExperiment:
    """One experiment output feeding position ``index`` of the shared stacks."""

    def __init__(self, stacks: Stacks, path: str, layers: list[str], index: int):
        self.stacks = stacks
        self.index = index
        self.exp = pyclm_io.open(path)
        self.requested = [_resolve_layer(self.exp, k) for k in layers]
        self.requested = [r for r in self.requested if r is not None]
        self.gkeys: dict[str, str] = {}
        for group, channel in self.requested:
            g = self.exp.groups[group]
            shape = g.shape if g.shape != (0, 0) else (1, 1)
            gkey = self.gkeys.setdefault(
                group, stacks.group_key(group, g.every_t, g.t_delay)
            )
            stacks.channel(gkey, channel, g.every_t, g.t_delay, shape)
            if self.exp.affine_transform is not None:
                stacks.pattern(gkey, g.every_t, g.t_delay, shape)
        self.refresh()

    @property
    def name(self) -> str:
        return self.exp.name

    def refresh(self) -> set[ChannelStack]:
        """Load the frames written since last time; returns the stacks that changed."""
        try:
            self.exp.refresh()
        except Exception as e:  # file being written, retry next tick
            logger.debug(f"refresh failed: {e}")
            return set()

        changed: set[ChannelStack] = set()
        for group, channel in self.requested:
            g = self.exp.groups[group]
            gkey = self.gkeys[group]
            stack = self.stacks.channel(gkey, channel, g.every_t, g.t_delay, (1, 1))
            for i in g.acquired():
                if (self.index, i) in stack.loaded:
                    continue
                frame = g.frame(i, channel)
                if frame is None:
                    continue
                stack.put(self.index, i, frame)
                changed.add(stack)

                if self.exp.affine_transform is not None:
                    pstack = self.stacks.pattern(gkey, g.every_t, g.t_delay, (1, 1))
                    if (self.index, i) not in pstack.loaded:
                        pat = self.exp.pattern_at(g.global_t(i))
                        if pat is not None:
                            overlay = pattern_to_camera(
                                pat, self.exp.affine_transform, frame.shape, g.binning
                            )
                            pstack.put(self.index, i, overlay)
        return changed

    def close(self):
        self.exp.close()


# =============================================================== controls
class ViewerControls(QtWidgets.QWidget):
    """
    The positions list (click to view), "follow the run", and the contrast
    controls, under the status line and minimap in the viewer's dock.
    """

    positionChosen = QtCore.Signal(int)
    normalizeRequested = QtCore.Signal()
    autoContrastChanged = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._row_changed)
        self.follow = QtWidgets.QCheckBox("Follow the run")
        self.follow.setToolTip(
            "Show the position the microscope is acquiring (from status.json)"
        )
        self.auto_contrast = QtWidgets.QCheckBox("Auto-contrast")
        self.auto_contrast.setChecked(True)
        self.auto_contrast.setToolTip(
            "Normalise each channel as frames arrive; turns off when you move a "
            "contrast slider"
        )
        self.auto_contrast.toggled.connect(self.autoContrastChanged)
        self.normalize = QtWidgets.QPushButton("Normalise now")
        self.normalize.setToolTip("Reset the contrast limits of every channel once")
        self.normalize.clicked.connect(self.normalizeRequested)
        self.hint = QtWidgets.QLabel("[ / ] : previous / next position")
        self.hint.setStyleSheet("color: gray")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QtWidgets.QLabel("Positions"))
        layout.addWidget(self.list, stretch=1)
        layout.addWidget(self.follow)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.auto_contrast)
        row.addWidget(self.normalize)
        layout.addLayout(row)
        layout.addWidget(self.hint)
        self._labels: list[str] = []

    def set_positions(self, labels: list[str]) -> None:
        self._labels = list(labels)
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems(self._labels)
        self.list.blockSignals(False)

    def select(self, index: int) -> None:
        """Reflect the viewed position without emitting positionChosen."""
        if 0 <= index < self.list.count() and self.list.currentRow() != index:
            self.list.blockSignals(True)
            self.list.setCurrentRow(index)
            self.list.blockSignals(False)

    def _row_changed(self, row: int) -> None:
        if row >= 0:
            self.positionChosen.emit(row)


class ViewerDock(QtWidgets.QWidget):
    """The status line, the minimap and the viewer controls in one dock."""

    def __init__(self, directory: Path | None, parent=None):
        super().__init__(parent)
        self.overview = RunOverview(directory)
        self.controls = ViewerControls()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.overview, stretch=2)
        layout.addWidget(self.controls, stretch=1)


# ================================================================== app
class ViewerApp:
    def __init__(
        self,
        specs: list[tuple[str, str]],
        status_path: Path | None = None,
        experiment_dir: Path | None = None,
        show: bool = True,
    ):
        self.viewer = napari.Viewer(show=show)
        self.status_path = status_path
        if experiment_dir is None and status_path is not None:
            experiment_dir = Path(status_path).parent
        self.dock = ViewerDock(experiment_dir)
        self.overview = self.dock.overview
        self.controls = self.dock.controls
        self.viewer.window.add_dock_widget(
            self.dock, name="Run", area="right", tabify=False
        )

        by_path: dict[str, list[str]] = {}
        for path, layer in specs:
            by_path.setdefault(path, []).append(layer)
        self.overview.refresh()
        order = self._position_order(list(by_path))
        self.stacks = Stacks(self.viewer)
        self.experiments = [
            LiveExperiment(self.stacks, path, by_path[path], index)
            for index, path in enumerate(order)
        ]
        self.positions = [e.name for e in self.experiments]
        self.controls.set_positions(self.positions)
        self.viewer.dims.axis_labels = AXIS_LABELS
        first = next(iter(self.stacks.channels.values()), None)
        if first is not None:
            self.viewer.layers.selection.active = first.layer
        self.normalize_all()

        # position selection: list, minimap, keys, slider -> all in sync
        self.controls.positionChosen.connect(self.view_position)
        self.overview.minimap.positionClicked.connect(self.view_label)
        self.controls.normalizeRequested.connect(self.normalize_all)
        self.controls.autoContrastChanged.connect(self._set_auto_contrast)
        self.viewer.dims.events.current_step.connect(self._slider_moved)
        self.viewer.bind_key("]", self.next_position, overwrite=True)
        self.viewer.bind_key("[", self.previous_position, overwrite=True)
        self.controls.select(0)

        self._fov_applied = False
        self._timer = QtCore.QTimer()
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        try:
            self.viewer.window._qt_window.destroyed.connect(lambda *_: self.close())
        except Exception:
            pass

    # ------------------------------------------------------- positions
    def _position_order(self, paths: list[str]) -> list[str]:
        """Outputs in the order of the position list (minimap), the rest after."""
        names = {}
        for path in paths:
            try:
                with pyclm_io.open(path) as exp:
                    names[path] = exp.name
            except Exception:
                names[path] = Path(path).stem
        listed = [p.label for p in self.overview.minimap.positions]
        rank = {label: i for i, label in enumerate(listed)}
        return sorted(paths, key=lambda p: (rank.get(names[p], len(rank)), names[p]))

    @property
    def current_position(self) -> int:
        return int(self.viewer.dims.current_step[0])

    def view_position(self, index: int) -> None:
        """Show position ``index``: the position slider, the list and the minimap follow."""
        if not 0 <= index < len(self.positions):
            return
        if self.current_position != index:
            self.viewer.dims.set_current_step(0, index)
        self.controls.select(index)

    def view_label(self, label: str) -> None:
        if label in self.positions:
            self.view_position(self.positions.index(label))

    def next_position(self, viewer=None) -> None:
        self.view_position((self.current_position + 1) % max(len(self.positions), 1))

    def previous_position(self, viewer=None) -> None:
        self.view_position((self.current_position - 1) % max(len(self.positions), 1))

    def _slider_moved(self, event=None) -> None:
        self.controls.select(self.current_position)

    # -------------------------------------------------------- contrast
    def _set_auto_contrast(self, on: bool) -> None:
        for stack in self.stacks.channels.values():
            stack.auto_contrast = on
        if on:
            self.normalize_all()

    def normalize_all(self) -> None:
        for stack in self.stacks.channels.values():
            stack.reset_contrast()

    # --------------------------------------------------------- refresh
    def refresh(self) -> int:
        status = self.overview.refresh()
        self.show_status()
        self._apply_fov()
        changed: set[ChannelStack] = set()
        for e in self.experiments:
            changed |= e.refresh()
        for stack in changed:
            if stack.auto_contrast:
                stack.reset_contrast()
        if self.controls.auto_contrast.isChecked() and not all(
            s.auto_contrast for s in self.stacks.channels.values()
        ):
            # a slider was moved: reflect it in the checkbox without re-triggering
            self.controls.auto_contrast.blockSignals(True)
            self.controls.auto_contrast.setChecked(False)
            self.controls.auto_contrast.blockSignals(False)
        if self.controls.follow.isChecked() and status:
            current = status.get("current_experiment")
            if current in self.positions:
                self.view_label(current)
        return len(changed)

    def _apply_fov(self):
        """Draw each position's field of view once the stores report a shape and pixel size."""
        if self._fov_applied or not self.overview.minimap.positions:
            return
        fov = {}
        for live in self.experiments:
            for g in live.exp.groups.values():
                if g.shape != (0, 0) and g.pixel_size_um:
                    fov[live.exp.name] = (
                        g.shape[0] * g.pixel_size_um,
                        g.shape[1] * g.pixel_size_um,
                    )
                    break
        if not fov:
            return
        for p in self.overview.minimap.positions:
            p.fov_um = fov.get(p.label)
        self.overview.minimap.redraw()
        self._fov_applied = True

    def show_status(self):
        """The status line (from status.json, written by the Manager) in napari's status bar."""
        try:
            self.viewer.status = self.overview.status.text()
        except Exception:
            pass

    def close(self):
        self._timer.stop()
        for e in self.experiments:
            e.close()

    def run(self):
        napari.run()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live napari viewer for PyCLM outputs")
    p.add_argument("experiment", help="directory containing experiment files")
    p.add_argument(
        "--src",
        action="append",
        type=_parse_src,
        help='Repeatable: "file.zarr:imaging/545"',
    )
    p.add_argument(
        "--every_t", default=0, type=int, help="ignored; kept for compatibility"
    )
    p.add_argument("--config", help="ignored; kept for compatibility", default=None)
    args = p.parse_args(argv)
    experiment_dir = Path(args.experiment)

    if args.src:
        specs = args.src
    else:
        layers_file = experiment_dir / "all_layers.txt"
        if layers_file.exists():
            specs = _load_layers_file(layers_file)
        else:
            specs = []
            for out in pyclm_io.find_experiments(experiment_dir):
                with pyclm_io.open(out) as exp:
                    for gname, g in exp.groups.items():
                        specs += [(str(out), f"{gname}/{c}") for c in g.channels]
    if not specs:
        raise SystemExit(f"no experiment outputs found in {experiment_dir}")

    ViewerApp(specs, experiment_dir / "status.json", experiment_dir).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
