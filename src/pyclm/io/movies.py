"""
Preview movies of a PyCLM run: one MP4 per experiment output (position).

Each movie frame is one imaging timepoint: the channels side by side in gray,
the light pattern in force at that timepoint as a translucent cyan overlay,
and segmentation outlines where the run saved them. Contrast is fixed per
channel over the whole movie (percentiles of a sample of frames), so changes
in brightness stay visible and the movie does not flicker. Frames are shown
as acquired: never cropped or rotated.

Encoding: ffmpeg (H.264) when it is on PATH or installed through
``imageio-ffmpeg``; otherwise OpenCV's H.264, then its MPEG-4 Part 2 encoder
(which plays in VLC but not in every Windows player).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .readers import ExperimentData, GroupData

logger = logging.getLogger(__name__)

PATTERN_BGR = np.array([255, 255, 0], np.float32)  # cyan, in OpenCV's BGR order
OUTLINE_BGR = (0, 215, 255)  # yellow-orange
PATTERN_ALPHA = 0.4


def contrast(frames: list[np.ndarray], low: float = 1.0, high: float = 99.8):
    """One display range for a channel over a whole movie, from a sample of its frames."""
    pick = frames[:: max(1, len(frames) // 12)]
    if not pick:
        return 0.0, 1.0
    v = np.concatenate([np.asarray(f, np.float32).ravel()[::7] for f in pick])
    lo, hi = np.percentile(v, [low, high])
    return float(lo), float(max(hi, lo + 1))


def to_gray(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.clip((np.asarray(frame, np.float32) - lo) / (hi - lo), 0, 1) * 255
    return np.repeat(g[..., None], 3, axis=2)


def outline_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels on the border of a labeled object (1 px wide, inside the object)."""
    lb = np.asarray(labels)
    edge = np.zeros(lb.shape, bool)
    edge[:-1] |= lb[:-1] != lb[1:]
    edge[1:] |= lb[1:] != lb[:-1]
    edge[:, :-1] |= lb[:, :-1] != lb[:, 1:]
    edge[:, 1:] |= lb[:, 1:] != lb[:, :-1]
    return edge & (lb > 0)


def compose(
    panel: np.ndarray, labels: np.ndarray | None, pattern: np.ndarray | None
) -> np.ndarray:
    """Gray panel (H, W, 3 float) + translucent cyan where lit + outlines on top; uint8 BGR."""
    out = panel.copy()
    if pattern is not None:
        lit = np.asarray(pattern, np.float32)[..., None] / 255.0 * PATTERN_ALPHA
        out = out * (1 - lit) + PATTERN_BGR * lit
    out = np.clip(out, 0, 255).astype(np.uint8)
    if labels is not None and labels.shape == out.shape[:2]:
        out[outline_mask(labels)] = OUTLINE_BGR
    return out


def _text(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    for color, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(
            img,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            width,
            cv2.LINE_AA,
        )
    return img


def imaging_group(exp: ExperimentData) -> GroupData | None:
    """The group with the most acquired frames (the stimulation group last)."""
    best, best_n = None, -1
    for name, g in exp.groups.items():
        n = len(g.acquired())
        if name.startswith("stim"):
            n = -0.5 if n else -1
        if n > best_n:
            best, best_n = g, n
    return best if best is not None and best.acquired() else None


def movie_frames(
    exp: ExperimentData,
    group: GroupData,
    channels: list[str] | None = None,
    scale: float = 1.0,
    pattern: bool = True,
    outlines: bool = True,
) -> Iterator[np.ndarray]:
    """BGR uint8 frames for one experiment, one per acquired slot of ``group``."""
    import cv2

    from .export import pattern_overlay

    slots = group.acquired()
    channels = [c for c in (channels or group.channels) if c in group.channels]
    if not channels:
        channels = list(group.channels)
    raw = {c: [group.frame(i, c) for i in slots] for c in channels}
    ranges = {c: contrast([f for f in raw[c] if f is not None]) for c in channels}
    seg_names = list(group.label_names) if outlines else []
    shape = group.shape
    for k, i in enumerate(slots):
        t = group.global_t(i)
        pat = None
        if pattern:
            try:
                pat = pattern_overlay(exp, group, shape, t)
            except Exception:
                pat = None
        panels = []
        for c in channels:
            f = raw[c][k]
            panel = to_gray(np.zeros(shape) if f is None else f, *ranges[c])
            lb = None
            for name in seg_names:
                lb = group.labels(i, c, name)
                if lb is not None:
                    break
            if lb is None and seg_names:
                # a segmentation belongs to one channel: draw it on every panel
                for other in channels:
                    lb = group.labels(i, other, seg_names[0])
                    if lb is not None:
                        break
            panels.append(compose(panel, lb, pat))
        img = np.concatenate(panels, axis=1)
        if scale != 1:
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        minutes = t * (group.interval_s or 60.0) / 60.0
        yield _text(
            np.ascontiguousarray(img),
            f"{exp.name}  t = {minutes:.0f} min  ({' | '.join(channels)})",
        )


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def write_mp4(frames: Iterable[np.ndarray], path: Path, fps: float = 10.0) -> str:
    """Write BGR frames to ``path`` (through a temporary name); returns the encoder used."""
    import cv2

    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ValueError("no frames")
    h, w = first.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.mp4")
    exe = _ffmpeg()
    if exe:
        cmd = [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w2}x{h2}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(tmp),
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in [first, *frames]:
            p.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
        p.stdin.close()
        if p.wait() == 0:
            tmp.replace(path)
            return "ffmpeg libx264"
        raise RuntimeError(f"ffmpeg failed for {path}")
    writer = None
    for code in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*code), fps, (w2, h2))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError(
            "no MP4 encoder available (install ffmpeg or imageio-ffmpeg)"
        )
    for f in [first, *frames]:
        writer.write(np.ascontiguousarray(f[:h2, :w2]))
    writer.release()
    tmp.replace(path)
    return f"opencv {code}"


def export_movies(
    exp: ExperimentData,
    out_dir: Path,
    channels: list[str] | None = None,
    scale: float = 1.0,
    fps: float = 10.0,
    pattern: bool = True,
    outlines: bool = True,
    split: bool = False,
) -> list[Path]:
    """
    Write ``<out_dir>/<experiment>.mp4`` (or one ``<experiment>_<channel>.mp4``
    per channel with ``split``). Returns the paths written; nothing if the
    experiment has no imaged frames.
    """
    group = imaging_group(exp)
    if group is None:
        logger.info(f"{exp.name}: nothing imaged, no movie")
        return []
    chans = [c for c in (channels or group.channels) if c in group.channels]
    if channels and not chans:
        return []
    chans = chans or list(group.channels)
    jobs = (
        [([c], f"{exp.name}_{c}.mp4") for c in chans]
        if split
        else [(chans, f"{exp.name}.mp4")]
    )
    written = []
    for cs, name in jobs:
        path = Path(out_dir) / name
        encoder = write_mp4(
            movie_frames(exp, group, cs, scale, pattern, outlines), path, fps
        )
        logger.info(f"{exp.name}: {path} ({encoder})")
        written.append(path)
    return written
