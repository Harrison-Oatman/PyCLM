"""
``pyclm preview``: run one experiment's segmentation(s) and pattern method
on one image and write what a user needs to judge them.

The machinery is the run's own (``SegmentationProcess``, ``TrackingProcess``,
``PatternProcess``, a ``DataDock`` and ``PatternContext``, the SLM buffer's
camera-to-DMD transform), so what preview shows is what the run does. The
image is a TIF (used for every channel the method needs) or one frame
snapped per channel from the microscope.

See docs/stage5-schema-setup-design.md §6.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .check import find_pyclm_config
from .core.datatypes import AcquisitionData, TrackingData
from .core.events import AcquisitionEvent
from .core.experiments import ImagingConfig, MicroscopePosition
from .core.grid import cut, geometry_for, stitch
from .core.kinds import is_seg_kind, seg_name
from .core.manager import SLMBuffer
from .core.pattern_process import PatternProcess
from .core.patterns import (
    ROI,
    CameraProperties,
    DataDock,
    PatternContext,
)
from .core.plan import _channel_group, _stim_channel_name
from .core.queues import AllQueues
from .core.segmentation_process import SegmentationProcess
from .core.tracking_process import TrackingProcess
from .directories import dry_settings_from_directory
from .schema import ConfigError, ExperimentConfig, PyclmConfig, ScheduleConfig

logger = logging.getLogger(__name__)


@dataclass
class PreviewResult:
    directory: Path
    label: str
    out_dir: Path
    paths: dict[str, Path] = field(default_factory=dict)
    pattern: np.ndarray | None = None
    requests: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def text(self) -> str:
        lines = [f"preview of {self.label} -> {self.out_dir}"]
        info = self.info
        lines.append(
            f"  pattern method '{info.get('method')}' {info.get('kwargs', {})}: "
            f"{info.get('lit_fraction', 0) * 100:.1f} % of the field lit, "
            f"generated in {info.get('generate_s', 0):.2f} s"
        )
        for name, seconds in info.get("segmentation_s", {}).items():
            lines.append(
                f"  segmentation '{name}': {info['objects'].get(name, 0)} objects in {seconds:.2f} s"
            )
        for change in self.requests:
            lines.append(f"  would change: {change}")
        for key, path in self.paths.items():
            lines.append(f"  {key}: {path.name}")
        return "\n".join(lines)


def _resolve_experiment(directory: Path, experiment: str) -> tuple[Path, str]:
    """(TOML path, position label) for a position label or a TOML stem."""
    stem = experiment.split(".")[0]
    toml = directory / f"{stem}.toml"
    if not toml.exists():
        raise ConfigError(toml, [f"no experiment file for '{experiment}'"])
    label = experiment if "." in experiment else f"{stem}.preview"
    return toml, label


def _position_for(directory: Path, label: str) -> MicroscopePosition:
    """The directory's position with this label (a grid keeps its tiles), else a stand-in."""
    from .check import _load_positions

    try:
        _name, positions = _load_positions(directory)
    except Exception:
        positions = None
    for pos in positions or ():
        if pos.label == label:
            return pos
    return MicroscopePosition(0.0, 0.0, 0.0, label=label)


def _load_image(path: Path) -> np.ndarray:
    import tifffile

    arr = np.asarray(tifffile.imread(str(path)))
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _apply_channel(core, cfg: ImagingConfig):
    for group, preset in cfg.get_config_groups():
        core.setConfig(group, preset)
    for dp in cfg.get_device_properties():
        core.setProperty(dp.device, dp.property, dp.value)
    core.setExposure(cfg.exposure)


def _overlay(raw: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(raw, (1, 99.5))
    grey = np.clip((raw.astype(float) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([grey, grey, grey], axis=-1)
    pat = np.clip(np.asarray(pattern, dtype=float), 0, 1)
    if pat.shape != grey.shape:
        from skimage.transform import resize

        pat = resize(pat, grey.shape, order=0, preserve_range=True)
    rgb[..., 1] = np.maximum(rgb[..., 1], pat)
    rgb[..., 2] = np.maximum(rgb[..., 2], pat)
    return (rgb * 255).astype(np.uint8)


def preview(
    directory,
    experiment: str,
    image=None,
    snap: bool = False,
    config_path=None,
    out_dir=None,
    t: int = 0,
    pixel_size_um: float | None = None,
    pattern_methods=None,
    segmentation_methods=None,
    tracking_methods=None,
) -> PreviewResult:
    """See the module docstring. Returns the written paths and the pattern."""
    import tifffile
    from skimage.io import imsave

    directory = Path(directory)
    toml, label = _resolve_experiment(directory, experiment)
    cfg = ExperimentConfig.from_file(toml)
    exp = cfg.to_experiment(label)
    stem = toml.stem

    interval_s = 0.0
    if (directory / "schedule.toml").exists():
        interval_s = ScheduleConfig.from_file(
            directory / "schedule.toml"
        ).timing.interval_seconds
    config = None
    found = find_pyclm_config(directory, config_path)
    if found is not None:
        config = PyclmConfig.from_file(found)

    if out_dir is None:
        out_dir = directory / "preview" / label
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = PreviewResult(directory, label, out_dir)

    # ---------------------------------------------------- image source
    core = None
    dry = dry_settings_from_directory(directory)
    source_binning = 1
    if image is not None:
        frame = _load_image(Path(image))
        # no pixel size given: the directory's dry_run.yml (or its defaults)
        # says what the TIFs are
        px = dry.pixel_size_um if pixel_size_um is None else float(pixel_size_um)
        source_binning = dry.binning

        def get_frame(channel_cfg: ImagingConfig) -> np.ndarray:
            return frame

        source = str(image)
    elif snap:
        if config is None:
            raise ConfigError(
                None,
                [
                    "--snap needs pyclm_config.toml (config_path to the MicroManager .cfg)"
                ],
            )
        from .core.real_core import RealMicroscopeCore

        core = RealMicroscopeCore()
        core.loadSystemConfiguration(config.config_path)
        if config.focus_device:
            core.setFocusDevice(config.focus_device)
        if config.camera_roi is not None:
            core.setROI(*[int(v) for v in config.camera_roi])
        px = (
            float(core.getPixelSizeUm())
            if pixel_size_um is None
            else float(pixel_size_um)
        )
        snapped: dict = {}

        def get_frame(channel_cfg: ImagingConfig) -> np.ndarray:
            key = channel_cfg.channel_id
            if key not in snapped:
                _apply_channel(core, channel_cfg)
                core.waitForSystem()
                core.snapImage()
                snapped[key] = np.asarray(core.getImage())
            return snapped[key]

        source = "snapped from the microscope"
    else:
        raise ValueError("preview needs an image (image=...) or snap=True")

    # ------------------------------------------------------ processes
    seg_proc = SegmentationProcess()
    for name, cls in (segmentation_methods or {}).items():
        seg_proc.register_method(cls, name)
    tracking_proc = TrackingProcess()
    for name, cls in (tracking_methods or {}).items():
        tracking_proc.register_method(cls, name)
    pp = PatternProcess(AllQueues())
    for name, cls in (pattern_methods or {}).items():
        pp.register_method(cls, name)

    binning = int(exp.stimulation.binning)
    # the image is at the imaging binning; the camera ROI is the unbinned size
    # the first imaging channel, or the stimulation channel of an experiment
    # that images nothing else
    if exp.channels:
        probe_channel = next(iter(exp.channels))
        probe_cfg = exp.channels[probe_channel]
    else:
        probe_channel = _stim_channel_name(exp, _channel_group(exp))
        probe_cfg = exp.stimulation
    probe = get_frame(probe_cfg)
    h, w = probe.shape

    # a grid position: the method sees the stitched frame. A tile-sized image
    # is tiled rows x columns; a stitched-size one is used as it is.
    position = _position_for(directory, label)
    geometry = None
    if position.is_grid:
        tile_h, tile_w = h * binning, w * binning
        stitched = geometry_for(position, (0, 0, tile_w, tile_h), px)
        if (h, w) == tuple(stitched.shape(binning)):
            # already stitched: the tile is the frame divided by the layout
            tile_h = (h * binning - (stitched.rows - 1) * stitched.pitch(1)[0]) // 1
            tile_w = (w * binning - (stitched.columns - 1) * stitched.pitch(1)[1]) // 1
            geometry = geometry_for(position, (0, 0, tile_w, tile_h), px)
        else:
            geometry = stitched
            tiles = [probe] * len(geometry.tiles)
            probe = stitch(tiles, geometry, binning)

            def get_frame(channel_cfg: ImagingConfig, _f=get_frame) -> np.ndarray:
                return stitch(
                    [_f(channel_cfg)] * len(geometry.tiles), geometry, binning
                )

            h, w = probe.shape
        position.geometry = geometry
        pp.grids = {label: geometry}
        tile_h, tile_w = geometry.tile_shape(1)
        pp.initialize(CameraProperties(ROI(0, 0, tile_w, tile_h), px / binning))
    else:
        pp.initialize(
            CameraProperties(ROI(0, 0, w * binning, h * binning), px / binning)
        )
    reqs = pp.request_method(exp)
    pp.initialize_models()
    model = pp.models[label]

    by_id = {c.channel_id: (name, c) for name, c in exp.channels.items()}
    by_id[exp.stimulation.channel_id] = ("stimulation", exp.stimulation)
    pp.positions = {label: position}

    dock = DataDock(t * interval_s, reqs)
    seg_seconds: dict[str, float] = {}
    objects: dict[str, int] = {}
    labels_by_channel: dict[tuple[str, str], np.ndarray] = {}
    frames: dict[str, np.ndarray] = {}
    for req in reqs:
        channel, channel_cfg = by_id[req.id]
        raw = get_frame(channel_cfg)
        frames[channel] = raw
        event = AcquisitionEvent(
            label,
            position,
            req.id,
            index={"t": t, "p": label, "c": channel},
            exposure_time_ms=channel_cfg.exposure,
            binning=channel_cfg.binning,
        )
        event.pixel_width_um = px
        data = AcquisitionData(event, raw)
        for kind in req.kinds:
            if kind == "raw":
                dock.add(data)
            elif is_seg_kind(kind):
                name = seg_name(kind)
                if (label, name) not in seg_proc.models:
                    seg_proc.request_method(exp, name)
                started = time.perf_counter()
                seg = seg_proc.run_model(label, data, name)
                seg_seconds[name] = (
                    seg_seconds.get(name, 0.0) + time.perf_counter() - started
                )
                lab = np.asarray(seg.data)
                objects[name] = len(np.unique(lab[lab > 0]))
                labels_by_channel[(channel, name)] = lab
                dock.add(seg)
            elif kind == "tracks":
                tracked = exp.tracking.segmentation
                if (channel, tracked) not in labels_by_channel:
                    if (label, tracked) not in seg_proc.models:
                        seg_proc.request_method(exp, tracked)
                    seg = seg_proc.run_model(label, data, tracked)
                    labels_by_channel[(channel, tracked)] = np.asarray(seg.data)
                method = tracking_proc.request_method(exp, channel)
                lab, rows = method.track(labels_by_channel[(channel, tracked)], t, px)
                dock.add(TrackingData(event, lab, rows))
                labels_by_channel[(channel, "tracks")] = lab
    if not dock.check_complete():
        raise RuntimeError(f"preview could not gather {dock.get_awaiting()}")
    if not frames:
        # an open-loop method asks for nothing; still show the image it would light
        frames[probe_channel] = probe

    # -------------------------------------------------------- pattern
    context = PatternContext(dock, exp, t=t, position=position)
    started = time.perf_counter()
    pattern = np.asarray(model.generate(context), dtype=np.float32)
    generate_s = time.perf_counter() - started
    result.pattern = pattern
    result.requests = [c.describe() for c in context.requests]

    # ---------------------------------------------------------- outputs
    for channel, raw in frames.items():
        path = out_dir / f"raw_{channel}.tif"
        tifffile.imwrite(path, np.asarray(raw))
        result.paths[f"raw {channel}"] = path
    for (channel, name), lab in labels_by_channel.items():
        path = out_dir / f"labels_{channel}_{name}.tif"
        tifffile.imwrite(path, lab.astype(np.uint32 if name == "tracks" else np.uint16))
        result.paths[f"labels {channel} {name}"] = path
    camera_pattern = np.clip(pattern, 0, 1)
    path = out_dir / "pattern_camera.tif"
    tifffile.imwrite(path, camera_pattern.astype(np.float32))
    result.paths["pattern (camera)"] = path
    first = next(iter(frames.values()), None)
    if first is not None:
        path = out_dir / "pattern_overlay.png"
        imsave(path, _overlay(first, camera_pattern), check_contrast=False)
        result.paths["overlay"] = path
    if config is not None:
        slm = SLMBuffer(AllQueues())
        slm.initialize(config.slm_shape, config.affine, [label], roi=config.camera_roi)
        if geometry is not None:
            # one DMD image per tile, stacked in tile order
            dmd = np.stack(
                [
                    slm.pattern_to_slm(tile, model.binning * source_binning)
                    for tile in cut(pattern, geometry, model.binning * source_binning)
                ]
            )
        else:
            dmd = slm.pattern_to_slm(pattern, model.binning * source_binning)
        path = out_dir / "pattern_dmd.tif"
        tifffile.imwrite(path, np.asarray(dmd, dtype=np.uint8))
        result.paths["pattern (DMD)"] = path
        dmd_lit = float((np.asarray(dmd) > 0).mean())
    else:
        dmd_lit = None
    lit = float((pattern > 0).mean()) if pattern.size else 0.0

    result.info = {
        "experiment": stem,
        "label": label,
        "t": t,
        "time_s": t * interval_s,
        "source": source,
        "pixel_size_um": px,
        "binning": binning,
        "source_binning": source_binning,
        "grid": None if geometry is None else geometry.as_dict(),
        "method": exp.pattern.method_name,
        "kwargs": exp.pattern.kwargs,
        "requirements": {by_id[r.id][0]: list(r.kinds) for r in reqs},
        "segmentation_s": seg_seconds,
        "objects": objects,
        "generate_s": generate_s,
        "lit_fraction": lit,
        "dmd_lit_fraction": dmd_lit,
        "requested_settings": result.requests,
        "outputs": {k: str(v) for k, v in result.paths.items()},
    }
    path = out_dir / "preview.json"
    path.write_text(json.dumps(result.info, indent=1, default=str), encoding="utf-8")
    result.paths["summary"] = path
    return result
