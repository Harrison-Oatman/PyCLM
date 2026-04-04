from __future__ import annotations

import os
import sys

os.environ.setdefault("NAPARI_DISABLE_PLUGIN_AUTOLOAD", "1")

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import napari
import numpy as np
from qtpy import QtCore


@dataclass
class LayerSpec:
    path: Path
    channel_key: str
    name: str | None = None


@dataclass
class ChannelSchedule:
    every_t: int = 1
    t_delay: int = 0
    t_stop: int = 0  # 0 means run to end
    t_count: int = 0

    def is_scheduled_at(self, t: int) -> bool:
        if t < self.t_delay:
            return False
        this_t = t - self.t_delay
        if self.t_stop > 0 and this_t >= self.t_stop:
            return False
        return this_t % self.every_t == 0


def _read_current_t_index(f: h5py.File) -> int:
    try:
        f["current_t_index"].id.refresh()
        return int(f["current_t_index"][()])
    except Exception:
        return -1


def _read_channel_schedule(f: h5py.File, channel_key: str) -> ChannelSchedule:
    sched = ChannelSchedule()
    try:
        every_t_map = json.loads(f.attrs.get("every_t", "{}"))
        sched.every_t = int(every_t_map.get(channel_key, 1))
    except Exception:
        pass
    try:
        sched.t_delay = int(f.attrs.get("t_delay", 0))
    except Exception:
        pass
    try:
        sched.t_stop = int(f.attrs.get("t_stop", 0))
    except Exception:
        pass
    try:
        sched.t_count = int(f.attrs.get("t_count", 0))
    except Exception:
        pass
    return sched


def _read_data_frame_swmr(
    f: h5py.File, t_val: str, channel_key: str
) -> np.ndarray | None:
    try:
        d = f[t_val][channel_key]["data"]
        d.id.refresh()
        arr = np.array(d)
        if arr.size == 0:
            print(f"t={t_val} no data")
            return None
        return arr
    except Exception:
        print(f"t = {t_val} exception")
        return None


def _upsample_to_absolute(
    frames: list[np.ndarray],
    acquired_at: list[int],
    t_count: int,
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    # Build a dense stack where each timepoint t holds the most recently acquired frame
    out = np.zeros(
        (t_count, *frame_shape), dtype=frames[0].dtype if frames else np.uint16
    )
    if not frames:
        return out
    acq_idx = 0
    for t in range(t_count):
        if acq_idx + 1 < len(acquired_at) and acquired_at[acq_idx + 1] <= t:
            acq_idx += 1
        if acquired_at[acq_idx] <= t:
            out[t] = frames[acq_idx]
    return out


class LiveHDF5Layer:
    def __init__(self, viewer: napari.Viewer, spec: LayerSpec):
        self.viewer = viewer
        self.spec = spec

        self.f: h5py.File | None = None
        self.last_t_index: int = -1
        self.frame_shape: tuple[int, ...] | None = None
        self.schedule: ChannelSchedule = ChannelSchedule()
        self._stack: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None

        if sys.platform != "win32":
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
        self.schedule = _read_channel_schedule(self.f, self.spec.channel_key)

    def _load_initial_stack(self) -> np.ndarray | None:
        if sys.platform == "win32":
            try:
                with h5py.File(
                    str(self.spec.path), mode="r", libver="latest", swmr=True
                ) as f:
                    self.schedule = _read_channel_schedule(f, self.spec.channel_key)
                    return self._do_load_initial(f)
            except (PermissionError, OSError, RuntimeError):
                return None
        else:
            if self.f is None:
                return None
            return self._do_load_initial(self.f)

    def _do_load_initial(self, f: h5py.File) -> np.ndarray | None:
        current_t = _read_current_t_index(f)
        if current_t < 0:
            return None

        frames: list[np.ndarray] = []
        acquired_at: list[int] = []
        for t in range(current_t + 1):
            if not self.schedule.is_scheduled_at(t):
                continue
            t_str = f"{t:05d}"
            frame = _read_data_frame_swmr(f, t_str, self.spec.channel_key)
            if frame is None:
                continue
            if self.frame_shape is None:
                self.frame_shape = frame.shape
            if frame.shape != self.frame_shape:
                continue
            frames.append(frame)
            acquired_at.append(t)

        self.last_t_index = current_t

        if not frames or self.frame_shape is None:
            return None

        self._last_frame = frames[-1]
        stack = _upsample_to_absolute(
            frames, acquired_at, current_t + 1, self.frame_shape
        )
        self._stack = stack
        return stack

    def refresh(self) -> bool:
        if sys.platform == "win32":
            try:
                with h5py.File(
                    str(self.spec.path), mode="r", libver="latest", swmr=True
                ) as f:
                    return self._do_refresh(f)
            except (PermissionError, OSError, RuntimeError):
                return False
        else:
            if self.f is None:
                self._open_file()
            if self.f is None:
                return False
            try:
                return self._do_refresh(self.f)
            except (PermissionError, OSError, RuntimeError):
                return False

    def _do_refresh(self, f: h5py.File) -> bool:
        current_t = _read_current_t_index(f)
        if current_t <= self.last_t_index:
            return False

        new_frames: list[np.ndarray] = []
        acquired_at: list[int] = []

        for t in range(self.last_t_index + 1, current_t + 1):
            if not self.schedule.is_scheduled_at(t):
                continue
            t_str = f"{t:05d}"
            frame = _read_data_frame_swmr(f, t_str, self.spec.channel_key)
            if frame is None:
                continue
            if self.frame_shape is None:
                self.frame_shape = frame.shape
            if frame.shape != self.frame_shape:
                continue
            new_frames.append(frame)
            acquired_at.append(t)

        n_new = current_t - self.last_t_index
        prev_last = self.last_t_index
        self.last_t_index = current_t

        if self.frame_shape is None:
            return False

        # Combine the previous last frame with new frames
        if self._last_frame is not None:
            hold_frames = [self._last_frame, *new_frames]
        else:
            hold_frames = new_frames
        if self._last_frame is not None:
            hold_at = [prev_last] + [a for a in acquired_at]
        else:
            hold_at = [a for a in acquired_at]

        dense_new = np.zeros((n_new, *self.frame_shape), dtype=np.uint16)
        if hold_frames:
            acq_idx = 0
            for i, t in enumerate(range(prev_last + 1, current_t + 1)):
                if acq_idx + 1 < len(hold_at) and hold_at[acq_idx + 1] <= t:
                    acq_idx += 1
                if hold_at[acq_idx] <= t:
                    dense_new[i] = hold_frames[acq_idx]

        if new_frames:
            self._last_frame = new_frames[-1]

        cur = self.layer.data

        if cur is None or cur.size == 0 or cur.shape == (1, 1, 1):
            self.layer.data = dense_new
            self.viewer.reset_view()
        else:
            if cur.shape[1:] != dense_new.shape[1:]:
                self.layer.data = dense_new
            else:
                self.layer.data = np.concatenate([cur, dense_new], axis=0)

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
