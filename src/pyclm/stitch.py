"""
``pyclm stitch``: one overview image of a run, each position's newest frame
placed where the stage was.

::

    pyclm stitch <dir>                      # once, to <dir>/stitched/overview.png
    pyclm stitch <dir> --every-s 60         # again every minute until the run is done
    pyclm stitch <dir> --channel 545 --orientation flip_ud

For every experiment output it takes the newest acquired frame of one channel
and the stage position and pixel size recorded with it in the frames table,
and draws the frame at that position on a downsampled canvas in stage
coordinates: columns follow stage +x, rows follow stage +y. Each field is
outlined and labeled. Contrast comes from fixed percentiles over all the
fields drawn, so fields can be compared.

How the camera image sits on the stage depends on the microscope. If the
fields come out mirrored or turned relative to each other, pass
``--orientation`` (``identity``, ``flip_ud``, ``flip_lr``, ``rot90``,
``rot180``, ``rot270``, ``transpose``): it is applied to every frame before
it is placed. A run of a few overlapping positions shows which one is right.

Read-only on the run: it writes ``stitched/overview.png`` (through a
temporary name, so a viewer never sees half a file) and nothing else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORIENTATIONS = {
    "identity": lambda a: a,
    "flip_ud": np.flipud,
    "flip_lr": np.fliplr,
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "transpose": np.transpose,
}


@dataclass
class Field:
    name: str
    image: np.ndarray  # oriented frame
    x_um: float  # stage position of the frame's center
    y_um: float
    pixel_size_um: float
    t: int


def _last(column, kinds, want="frame"):
    for v, k in zip(reversed(column), reversed(kinds), strict=True):
        if k == want and v is not None:
            return v
    return None


def newest_fields(
    directory: Path,
    channel: str | None = None,
    orientation: str = "identity",
    pixel_size_um: float | None = None,
) -> tuple[list[Field], list[str]]:
    """The newest frame of every experiment output, with where it was taken; and what was left out."""
    from .io import find_experiments
    from .io import open as open_experiment
    from .io.movies import imaging_group

    turn = ORIENTATIONS[orientation]
    fields, missing = [], []
    for path in find_experiments(directory):
        try:
            exp = open_experiment(path)
        except Exception as e:
            missing.append(f"{path.name}: cannot open ({e})")
            continue
        try:
            group = imaging_group(exp)
            if group is None:
                missing.append(f"{exp.name}: nothing imaged yet")
                continue
            ch = channel if channel in group.channels else group.channels[0]
            if channel and channel not in group.channels:
                missing.append(f"{exp.name}: no channel {channel}, used {ch}")
            slots = group.acquired()
            frame = None
            for i in reversed(slots):
                frame = group.frame(i, ch)
                if frame is not None:
                    t = group.global_t(i)
                    break
            if frame is None:
                missing.append(f"{exp.name}: no frame of {ch} yet")
                continue
            table = exp.frames
            x = y = px = None
            if table is not None and table.num_rows:
                cols = table.column_names
                kinds = table.column("kind").to_pylist() if "kind" in cols else []
                if "x" in cols and "y" in cols and kinds:
                    x = _last(table.column("x").to_pylist(), kinds)
                    y = _last(table.column("y").to_pylist(), kinds)
                if "pixel_size_um" in cols and kinds:
                    px = _last(table.column("pixel_size_um").to_pylist(), kinds)
            px = px or group.pixel_size_um or pixel_size_um
            if x is None or y is None or not px:
                missing.append(f"{exp.name}: no stage position or pixel size recorded")
                continue
            fields.append(
                Field(
                    exp.name, turn(np.asarray(frame)), float(x), float(y), float(px), t
                )
            )
        finally:
            exp.close()
    return fields, missing


def overview(fields: list[Field], max_px: int = 3000, title: str = "") -> np.ndarray:
    """A BGR uint8 canvas with every field drawn at its stage position."""
    import cv2

    if not fields:
        raise ValueError("no fields to draw")
    boxes = []
    for f in fields:
        h_um = f.image.shape[0] * f.pixel_size_um
        w_um = f.image.shape[1] * f.pixel_size_um
        boxes.append((f.x_um - w_um / 2, f.y_um - h_um / 2, w_um, h_um))
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    pad = 0.02 * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    um_per_px = max((x1 - x0) / max_px, (y1 - y0) / max_px, 1e-9)
    # never upsample: at most one frame pixel per canvas pixel
    um_per_px = max(um_per_px, min(f.pixel_size_um for f in fields))
    W = int(np.ceil((x1 - x0) / um_per_px)) + 1
    H = int(np.ceil((y1 - y0) / um_per_px)) + 1
    values = np.concatenate(
        [np.asarray(f.image, np.float32).ravel()[::17] for f in fields]
    )
    lo, hi = np.percentile(values, [1, 99.8])
    hi = max(hi, lo + 1)
    canvas = np.zeros((H, W, 3), np.uint8)
    for f, (bx, by, bw, bh) in zip(fields, boxes, strict=True):
        c0 = round((bx - x0) / um_per_px)
        r0 = round((by - y0) / um_per_px)
        w = max(1, round(bw / um_per_px))
        h = max(1, round(bh / um_per_px))
        small = cv2.resize(
            np.asarray(f.image, np.float32), (w, h), interpolation=cv2.INTER_AREA
        )
        gray = (np.clip((small - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        r1, c1 = min(H, r0 + h), min(W, c0 + w)
        canvas[r0:r1, c0:c1] = gray[: r1 - r0, : c1 - c0, None]
        cv2.rectangle(canvas, (c0, r0), (c1 - 1, r1 - 1), (255, 255, 0), 1)
        cv2.putText(
            canvas,
            f.name,
            (c0 + 4, r0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if title:
        cv2.putText(
            canvas,
            title,
            (8, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def stitch_once(
    directory: Path,
    out: Path | None = None,
    channel: str | None = None,
    orientation: str = "identity",
    max_px: int = 3000,
    pixel_size_um: float | None = None,
    log=print,
) -> Path | None:
    """Write the overview once; returns its path, or None if nothing could be drawn."""
    import cv2

    directory = Path(directory)
    out = Path(out) if out else directory / "stitched"
    fields, missing = newest_fields(directory, channel, orientation, pixel_size_um)
    for m in missing:
        log(f"  {m}")
    if not fields:
        log("nothing to stitch yet")
        return None
    stamp = time.strftime("%H:%M")
    img = overview(fields, max_px, f"{len(fields)} field(s), newest frames, {stamp}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "overview.png"
    tmp = out / "overview.part.png"
    cv2.imwrite(str(tmp), img)
    tmp.replace(path)
    log(f"{len(fields)} field(s) -> {path}")
    return path


def stitch(
    directory: Path,
    every_s: float | None = None,
    **kw,
) -> int:
    """Stitch once, or every ``every_s`` until ``status.json`` says done (then once more)."""
    from .watch import read_status

    if every_s is None:
        return 0 if stitch_once(directory, **kw) else 1
    while True:
        status, _ = read_status(Path(directory))
        stitch_once(directory, **kw)
        if status and status.get("done"):
            return 0
        time.sleep(every_s)
