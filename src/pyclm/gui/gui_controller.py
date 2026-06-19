from __future__ import annotations

import os
import sys

os.environ.setdefault("NAPARI_DISABLE_PLUGIN_AUTOLOAD", "1")

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import napari
import numpy as np
import pyqtgraph as pg
from h5py import File
from qtpy import QtCore
from qtpy.QtWidgets import QPlainTextEdit
from skimage.transform import downscale_local_mean
from toml import load


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
    f: h5py.File, t_val: str, channel_key: str, at, logger=None
) -> np.ndarray | None:
    if channel_key == "stim_dmd":
        return _read_stim_frame_swmr(f, t_val, at, logger)
    else:
        try:
            d = f[t_val][channel_key]["data"]
            d.id.refresh()
            arr = np.array(d)
            if logger:
                logger.appendPlainText(
                    f"{datetime.now().strftime('%H:%M:%S')} - Loaded data from {Path(f.filename).name} :: {channel_key} "
                    f"at timepoint {t_val}"
                )
            if arr.size == 0:
                print(f"t={t_val} no data")
                return None
            return arr
        except Exception:
            print(f"t = {t_val} exception")
            return None


def _read_stim_frame_swmr(
    f: h5py.File, t_val: str, at, logger=None
) -> np.ndarray | None:
    ati = cv2.invertAffineTransform(at)
    try:
        if t_val not in f:
            return None
        if "stim_aq" not in f[t_val].keys() or "dmd" not in f[t_val]["stim_aq"]:
            return None
        imaging_key = next(
            (k for k in f[t_val].keys() if k.startswith("channel_")), None
        )
        if imaging_key is None:
            return None
        d = f[t_val][imaging_key]["data"]
        d.id.refresh()
        data_shape = d.shape
        if 0 in data_shape:
            return None
        b = int(get_binning_from_metadata(f, imaging_key))
        pattern = np.array(f[t_val]["stim_aq"]["dmd"])
        tf = cv2.warpAffine(
            np.round(pattern).astype(np.uint8),
            ati,
            (data_shape[1] * b, data_shape[0] * b),
        ).astype(np.uint16)
        if logger:
            logger.appendPlainText(
                f"{datetime.now().strftime('%H:%M:%S')} - Loaded stim data from {Path(f.filename).name} "
                f"at timepoint {t_val}"
            )
        return downscale_local_mean(tf, (b, b)).astype(np.uint16)
    except Exception as e:
        print(f"t = {t_val} stim frame exception: {type(e).__name__}: {e}")
        return None


def _upsample_to_absolute(
    frames: list[np.ndarray],
    acquired_at: list[int],
    t_count: int,
    frame_shape: tuple[int, ...],
    hold: bool = True,
) -> np.ndarray:
    out = np.zeros(
        (t_count, *frame_shape), dtype=frames[0].dtype if frames else np.uint16
    )
    if not frames:
        return out
    if hold:
        acq_idx = 0
        for t in range(t_count):
            if acq_idx + 1 < len(acquired_at) and acquired_at[acq_idx + 1] <= t:
                acq_idx += 1
            if acquired_at[acq_idx] <= t:
                out[t] = frames[acq_idx]
    else:
        for frame, t in zip(frames, acquired_at, strict=False):
            out[t] = frame
    return out


class LiveHDF5Layer:
    def __init__(
        self,
        viewer: napari.Viewer,
        spec: LayerSpec,
        logger: QPlainTextEdit,
        at: np.ndarray | None = None,
    ):
        self.viewer = viewer
        self.spec = spec
        self.logger = logger

        self.at = at
        self.f: h5py.File | None = None
        self.last_t_index: int = -1
        self.frame_shape: tuple[int, ...] | None = None
        self.schedule: ChannelSchedule = ChannelSchedule()
        self._stack: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None
        self._hold: bool = spec.channel_key != "stim_dmd"

        if sys.platform != "win32":
            self._open_file()

        layer_name = spec.name or f"{spec.path.name} :: {spec.channel_key}"
        initial = self._load_initial_stack()
        if initial is None:
            initial = np.zeros((1, 1, 1), dtype=np.uint16)
        print(f"add_image: {layer_name} shape={initial.shape}")
        self.layer = self.viewer.add_image(initial, name=layer_name)
        if spec.channel_key == "stim_dmd":
            self.layer.colormap = "cyan"
            self.layer.contrast_limits = (0, 255)
            self.layer.contrast_limits_range = (0, 255)
            self.layer.opacity = 0.2
        else:
            self.layer.reset_contrast_limits()

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
            frame = _read_data_frame_swmr(
                f, t_str, self.spec.channel_key, self.at, logger=self.logger
            )
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
            frames, acquired_at, current_t + 1, self.frame_shape, hold=self._hold
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
            frame = _read_data_frame_swmr(
                f, t_str, self.spec.channel_key, self.at, logger=self.logger
            )
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

        dense_new = np.zeros((n_new, *self.frame_shape), dtype=np.uint16)
        if self._hold:
            hold_frames = (
                [self._last_frame] if self._last_frame is not None else []
            ) + new_frames
            hold_at = (
                [prev_last] if self._last_frame is not None else []
            ) + acquired_at
            if hold_frames:
                acq_idx = 0
                for i, t in enumerate(range(prev_last + 1, current_t + 1)):
                    if acq_idx + 1 < len(hold_at) and hold_at[acq_idx + 1] <= t:
                        acq_idx += 1
                    if hold_at[acq_idx] <= t:
                        dense_new[i] = hold_frames[acq_idx]
        else:
            for frame, t in zip(new_frames, acquired_at, strict=False):
                dense_new[t - (prev_last + 1)] = frame

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

        if self.spec.channel_key != "stim_dmd":
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
    def __init__(self, specs: Sequence[LayerSpec], at: np.ndarray | None = None):
        self.viewer = napari.Viewer()
        self._log_label = QPlainTextEdit()
        self._log_label.setReadOnly(True)

        # self._position_map = pg.PlotWidget(title="Position Map")
        # self._scatter_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200))
        # self._position_map.addItem(self._scatter_item)

        self.viewer.window.add_dock_widget(
            self._log_label, name="Logs", area="right", tabify=True
        )
        # self.viewer.window.add_dock_widget(
        #     self._position_map, name="Position Map", area="right", tabify=True
        # )

        self.layers = [
            LiveHDF5Layer(self.viewer, s, logger=self._log_label, at=at)
            for s in specs[::-1]
        ]

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


def _stim_exposure_nonzero(experiment_dir: Path, hdf5_path: Path) -> bool:
    stem = hdf5_path.stem.split(".")[0]
    toml_path = experiment_dir / f"{stem}.toml"
    if not toml_path.exists():
        return True
    try:
        data = load(toml_path)
        return data.get("stimulation", {}).get("exposure", 0) != 0
    except Exception:
        return True


def launch_hdf5_layer_viewer(
    specs: Sequence[tuple[str, str]],
    experiment_dir: Path | None = None,
    at: np.ndarray | None = None,
) -> HDF5LayerViewerApp:
    filtered = []

    seen_fps = set()

    for fp, ch in specs:
        if fp not in seen_fps:
            if _stim_exposure_nonzero(experiment_dir, Path(fp)):
                filtered.append((fp, "stim_dmd"))

        filtered.append((fp, ch))
        seen_fps.add(fp)

    layer_specs = [LayerSpec(path=Path(fp), channel_key=ch) for fp, ch in filtered]
    return HDF5LayerViewerApp(layer_specs, at=at)


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


def _parse_all_layers(s: Path) -> list[tuple[str, str]]:
    layers = []
    with open(s, encoding="utf-8") as file:
        for line in file:
            layers.append(_parse_src(line))
    return layers


def find_affine_transform(input_dir, config_path):
    # copied from main.py
    # search for config file if not provided
    if config_path is None:
        # look in the experiment directory for pyclm_config.toml
        config_path = input_dir / "pyclm_config.toml"

        # look in the current working directory for pyclm_config.toml
        if not config_path.exists():
            config_path = Path("pyclm_config.toml")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. Affine transform is required."
        )

    config = load(config_path)
    return np.array(config["affine_transform"], dtype=np.float32)


def get_binning_from_metadata(f: File, chan_key: str):
    """
    Extract binning from file attributes or return default 1.
    """
    if "experiment_metadata" in list(f.attrs.keys()):
        try:
            meta = json.loads(f.attrs["experiment_metadata"])
            # chan_key is typically "channel_NAME"
            # channel keys in metadata are "NAME"
            if chan_key.startswith("channel_"):
                short_name = chan_key.replace("channel_", "", 1)
                if short_name in meta.get("channels", {}):
                    return meta["channels"][short_name].get("binning", 1)
            else:
                return meta["segmentation"].get("binning", 1)
        except Exception as e:
            print(f"Error reading binning from metadata: {e}")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Napari SWMR HDF5 viewer (one layer per file/channel)"
    )
    p.add_argument("experiment", help="directory containing experiment files")
    p.add_argument(
        "--src",
        action="append",
        type=_parse_src,
        required=False,
        help='Repeatable: "file.hdf5:channel_638"',
    )
    p.add_argument("--config", help="path to pyclm_config.toml file", default=None)
    args = p.parse_args(argv)
    experiment_dir = Path(args.experiment)
    at = find_affine_transform(experiment_dir, args.config)
    if not args.src:
        all_layers_path = experiment_dir / "all_layers.txt"
        assert all_layers_path.exists(), (
            "no all_layers.txt found in experiment directory"
        )

        layers = _parse_all_layers(all_layers_path)

        app = launch_hdf5_layer_viewer(layers, experiment_dir=experiment_dir, at=at)
    else:
        app = launch_hdf5_layer_viewer(args.src, experiment_dir=experiment_dir, at=at)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
