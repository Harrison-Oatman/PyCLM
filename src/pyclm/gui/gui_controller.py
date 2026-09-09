"""
Live napari viewer for a running (or finished) PyCLM experiment directory.

Reads every output through ``pyclm.io`` so it works for both the HDF5 (format 1)
and OME-Zarr (format 2) layouts. Each cadence group becomes one layer per
channel with the layer's time scale set to the group's ``every_t``, so groups
of different cadence line up on the plan's time axis without blank frames.
A cyan overlay shows the DMD pattern in force at each frame when the affine
transform is known.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("NAPARI_DISABLE_PLUGIN_AUTOLOAD", "1")

import argparse
import json
import logging
from pathlib import Path

import napari
import numpy as np
from qtpy import QtCore

from pyclm import io as pyclm_io
from pyclm.io.export import pattern_to_camera

from .widgets import RunOverview

logger = logging.getLogger(__name__)


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


class LiveExperiment:
    """All napari layers for one experiment output, refreshed by polling."""

    def __init__(self, viewer: napari.Viewer, path: str, layers: list[str]):
        self.viewer = viewer
        self.exp = pyclm_io.open(path)
        self.layers: dict[tuple[str, str], napari.layers.Image] = {}
        self.loaded: dict[str, set[int]] = {}
        self.pattern_layers: dict[str, napari.layers.Image] = {}
        self.requested = [_resolve_layer(self.exp, k) for k in layers]
        self.requested = [r for r in self.requested if r is not None]

        for group, channel in self.requested:
            g = self.exp.groups[group]
            shape = g.shape if g.shape != (0, 0) else (1, 1)
            data = np.zeros((max(g.timepoints, 1), *shape), dtype=np.uint16)
            layer = self.viewer.add_image(
                data,
                name=f"{self.exp.name} :: {group}/{channel}",
                scale=(g.every_t, 1, 1),
                translate=(g.t_delay, 0, 0),
            )
            self.layers[(group, channel)] = layer
            self.loaded.setdefault(group, set())

        if self.exp.affine_transform is not None:
            for group in {g for g, _ in self.requested}:
                g = self.exp.groups[group]
                shape = g.shape if g.shape != (0, 0) else (1, 1)
                layer = self.viewer.add_image(
                    np.zeros((max(g.timepoints, 1), *shape), dtype=np.uint16),
                    name=f"{self.exp.name} :: {group}/pattern",
                    scale=(g.every_t, 1, 1),
                    translate=(g.t_delay, 0, 0),
                    colormap="cyan",
                    opacity=0.25,
                    contrast_limits=(0, 255),
                    blending="additive",
                )
                self.pattern_layers[group] = layer

        self.refresh()

    def refresh(self) -> bool:
        try:
            self.exp.refresh()
        except Exception as e:  # file being written, retry next tick
            logger.debug(f"refresh failed: {e}")
            return False

        changed = False
        for group, channel in self.requested:
            g = self.exp.groups[group]
            layer = self.layers[(group, channel)]
            for i in g.acquired():
                if (i, channel) in self.loaded[group]:
                    continue
                frame = g.frame(i, channel)
                if frame is None:
                    continue
                data = layer.data
                if data.shape[1:] != frame.shape:
                    data = np.zeros(
                        (max(g.timepoints, 1), *frame.shape), dtype=np.uint16
                    )
                if i >= data.shape[0]:
                    data = np.concatenate(
                        [
                            data,
                            np.zeros((i + 1 - data.shape[0], *frame.shape), np.uint16),
                        ]
                    )
                data[i] = frame
                layer.data = data
                self.loaded[group].add((i, channel))
                changed = True

                pattern_layer = self.pattern_layers.get(group)
                if pattern_layer is not None:
                    pat = self.exp.pattern_at(g.global_t(i))
                    if pat is not None:
                        overlay = pattern_to_camera(
                            pat, self.exp.affine_transform, frame.shape, g.binning
                        )
                        pdata = pattern_layer.data
                        if pdata.shape[1:] != overlay.shape or i >= pdata.shape[0]:
                            new = np.zeros(
                                (max(i + 1, pdata.shape[0]), *overlay.shape), np.uint16
                            )
                            new[: min(pdata.shape[0], new.shape[0])] = (
                                pdata[: new.shape[0]]
                                if pdata.shape[1:] == overlay.shape
                                else 0
                            )
                            pdata = new
                        pdata[i] = overlay
                        pattern_layer.data = pdata
            if changed:
                layer.reset_contrast_limits()
        return changed

    def close(self):
        self.exp.close()


class ViewerApp:
    def __init__(
        self,
        specs: list[tuple[str, str]],
        status_path: Path | None = None,
        experiment_dir: Path | None = None,
    ):
        self.viewer = napari.Viewer()
        self.status_path = status_path
        if experiment_dir is None and status_path is not None:
            experiment_dir = Path(status_path).parent
        self.overview = RunOverview(experiment_dir)
        self.viewer.window.add_dock_widget(
            self.overview, name="Run", area="right", tabify=False
        )
        by_path: dict[str, list[str]] = {}
        for path, layer in specs:
            by_path.setdefault(path, []).append(layer)
        self.experiments = [
            LiveExperiment(self.viewer, path, layers)
            for path, layers in by_path.items()
        ]

        self._fov_applied = False
        self._timer = QtCore.QTimer()
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        try:
            self.viewer.window._qt_window.destroyed.connect(lambda *_: self.close())
        except Exception:
            pass

    def refresh(self):
        self.show_status()
        self.overview.refresh()
        self._apply_fov()
        return sum(1 for e in self.experiments if e.refresh())

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
        """One line from status.json (written by the Manager every timepoint) in the status bar."""
        if self.status_path is None or not self.status_path.exists():
            return
        try:
            status = json.loads(self.status_path.read_text())
        except Exception:
            return
        parts = [f"t {status.get('t')}/{status.get('timepoints')}"]
        if status.get("done"):
            parts.append("done")
        late = [
            f"{name} late {info['lateness_s']:.1f}s"
            for name, info in status.get("experiments", {}).items()
            if info.get("lateness_s")
        ]
        errors = sum(
            info.get("errors", 0) for info in status.get("experiments", {}).values()
        )
        if late:
            parts += late
        if errors:
            parts.append(f"{errors} acquisition errors")
        if status.get("settings_applied"):
            parts.append(f"{status['settings_applied']} settings changed")
        try:
            self.viewer.status = " | ".join(parts)
        except Exception:
            pass

    def close(self):
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
