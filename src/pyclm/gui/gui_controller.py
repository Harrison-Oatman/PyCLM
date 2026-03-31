from __future__ import annotations

import os

os.environ.setdefault("NAPARI_DISABLE_PLUGIN_AUTOLOAD", "1")

import argparse
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import napari
import numpy as np
from natsort import natsorted
from qtpy import QtCore


@dataclass
class LayerSpec:
    path: Path
    channel_key: str
    name: str | None = None


def _read_current_t_index(f: h5py.File) -> int:
    try:
        f["current_t_index"].id.refresh()
        return int(f["current_t_index"][()])
    except Exception:
        return -1


def _read_data_frame_swmr(
    f: h5py.File, t_val: str, channel_key: str
) -> np.ndarray | None:
    try:
        d = f[t_val][channel_key]["data"]
        arr = np.array(d)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


class LiveHDF5Layer:
    def __init__(self, viewer: napari.Viewer, spec: LayerSpec):
        self.viewer = viewer
        self.spec = spec

        self.f: h5py.File | None = None
        self.last_t_index: int = -1
        self.frame_shape: tuple[int, ...] | None = None

        self._open_file()

        layer_name = spec.name or f"{spec.path.name} :: {spec.channel_key}"
        initial = self._load_initial_stack()
        if initial is None:
            initial = np.zeros((1, 1, 1), dtype=np.uint16)
        self.layer = self.viewer.add_image(initial, name=layer_name)

    def _open_file(self) -> None:
        try:
            if self.f is not None:
                self.f.close()
        except Exception:
            pass

        if not self.spec.path.exists():
            self.f = None
            return

        self.f = h5py.File(str(self.spec.path), mode="r", libver="latest", swmr=True)

    def _load_initial_stack(self) -> np.ndarray | None:
        if self.f is None:
            return None

        current_t = _read_current_t_index(self.f)
        if current_t < 0:
            return None

        frames: list[np.ndarray] = []
        for t in range(current_t + 1):
            t_str = f"{t:05d}"
            frame = _read_data_frame_swmr(self.f, t_str, self.spec.channel_key)
            if frame is None:
                continue
            if self.frame_shape is None:
                self.frame_shape = frame.shape
            if frame.shape != self.frame_shape:
                continue
            frames.append(frame)

        self.last_t_index = current_t
        return np.stack(frames, axis=0) if frames else None

    def refresh(self) -> bool:
        if self.f is None:
            self._open_file()
        if self.f is None:
            return False

        current_t = _read_current_t_index(self.f)
        if current_t <= self.last_t_index:
            return False

        new_frames: list[np.ndarray] = []
        for t in range(self.last_t_index + 1, current_t + 1):
            t_str = f"{t:05d}"
            frame = _read_data_frame_swmr(self.f, t_str, self.spec.channel_key)
            if frame is None:
                continue

            if self.frame_shape is None:
                self.frame_shape = frame.shape
            if frame.shape != self.frame_shape:
                continue

            new_frames.append(frame)

        self.last_t_index = current_t

        if not new_frames:
            return False

        new_stack = np.stack(new_frames, axis=0)
        cur = self.layer.data

        if cur is None or cur.size == 0 or cur.shape == (1, 1, 1):
            self.layer.data = new_stack
            self.viewer.reset_view()
        else:
            if cur.shape[1:] != new_stack.shape[1:]:
                self.layer.data = new_stack
            else:
                self.layer.data = np.concatenate([cur, new_stack], axis=0)

        self.layer.reset_contrast_limits()
        self.layer.refresh()
        return True

    def close(self) -> None:
        try:
            if self.f is not None:
                self.f.close()
        except Exception:
            pass


class HDF5LayerViewerApp:
    def __init__(self, specs: Sequence[LayerSpec]):
        self.viewer = napari.Viewer()
        self.layers = [LiveHDF5Layer(self.viewer, s) for s in specs]

        self._poll_timer = QtCore.QTimer()
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self.refresh)
        self._poll_timer.start()

        try:
            self.viewer.window._qt_window.destroyed.connect(lambda *_: self.close())
        except Exception:
            pass

    def refresh(self) -> int:
        changed = 0
        for layer in self.layers:
            if layer.refresh():
                changed += 1
        return changed

    def close(self) -> None:
        for layer in self.layers:
            layer.close()

    def run(self) -> None:
        napari.run()

    def show(self) -> None:
        self.viewer.window.show()


def launch_hdf5_layer_viewer(specs: Sequence[tuple[str, str]]) -> HDF5LayerViewerApp:
    layer_specs = [LayerSpec(path=Path(fp), channel_key=ch) for fp, ch in specs]
    return HDF5LayerViewerApp(layer_specs)


def _parse_src(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            'Each --src must be in the form "path:channel_638"'
        )
    path, ch = s.rsplit(":", 1)
    path = path.strip().strip('"')
    ch = ch.strip()
    if not path or not ch:
        raise argparse.ArgumentTypeError(
            'Each --src must be in the form "path:channel_638"'
        )
    return path, ch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Napari SWMR HDF5 viewer (one layer per file/channel)"
    )
    p.add_argument(
        "--src",
        action="append",
        type=_parse_src,
        required=True,
        help='Repeatable: "file.hdf5:channel_638"',
    )
    args = p.parse_args(argv)

    app = launch_hdf5_layer_viewer(args.src)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
