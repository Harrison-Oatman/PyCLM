"""
``pyclm new <dir> --demo``: an experiment directory that runs on any
computer, without a microscope or downloaded data.

It holds two experiments on the virtual microscope:

- ``bar``: open loop, a bar of light sweeping across the field;
- ``cells``: closed loop, cells found by a threshold segmentation that the
  directory brings with it (``demo_methods.py``, loaded through ``methods``
  in ``pyclm_config.toml``), lit on their outward-facing half (``move_out``).

The images are synthetic cells drawn here, so nothing is downloaded; the
configuration's DMD calibration covers the whole simulated camera. Run it
with ``pyclm run <dir> --dry`` (add ``--gui`` with the gui extra).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .templates import CLOSED_LOOP, OPEN_LOOP

FRAMES = 10
SIZE = 512  # pixels of the demo TIFs (4x binned: the simulated camera is 2048 square)
PIXEL_SIZE_UM = 0.325  # unbinned: 1.3 um per TIF pixel at the experiments' binning 4

SCHEDULE = """# The demo's timing: ten timepoints three seconds apart
[timing]
steps = 10
interval_seconds = 3
setup_time_seconds = 0.5
time_between_positions = 0.2
"""

CONFIG = """# The demo's configuration for the virtual microscope (pyclm run --dry).
# config_path is only read on a real microscope.
config_path = "C:\\\\Program Files\\\\Micro-Manager-2.0\\\\MMConfig.cfg"

# a calibration that maps the whole simulated camera (2048 x 2048 unbinned
# pixels) onto the DMD; a real one comes from the Projector plugin
affine_transform = [[0.4453125, 0.0, 0.0], [0.0, 0.556640625, 0.0]]
slm_shape_h = 1140
slm_shape_w = 912

settle_time_seconds = 0.0

# methods of this directory: the threshold segmentation cells.toml uses
methods = ["demo_methods.py"]

[output]
format = "ome-zarr"
pattern_policy = "on_change"
export_imagej = true
"""

DRY_RUN = f"""# What the virtual microscope shows at each position. The simulated camera
# is 2048 pixels square; the TIFs are its frames at binning 4, which the
# experiment TOMLs ask for, as they would on a real camera.
pixel_size_um: {PIXEL_SIZE_UM}   # unbinned camera pixel size
positions:
  - name: bar.00
    x: 0.0
    y: 0.0
    source: bar.tif
  - name: cells.00
    x: 2000.0
    y: 0.0
    source: cells.tif
"""

METHODS = '''"""
The demo's own segmentation, loaded through ``methods`` in pyclm_config.toml.

A method of your own looks like this: a subclass with a ``name`` (what the
TOML's ``method = ...`` selects) and keyword arguments the TOML can set.
"""

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects

from pyclm import SegmentationMethod


class ThresholdSegmentation(SegmentationMethod):
    """Cells are the connected regions brighter than Otsu's threshold."""

    name = "threshold"

    def __init__(self, experiment_name, min_size_px=20, **kwargs):
        super().__init__(experiment_name, **kwargs)
        self.min_size_px = int(min_size_px)

    def segment(self, data: np.ndarray) -> np.ndarray:
        image = np.asarray(data, dtype=np.float32)
        mask = image > threshold_otsu(image)
        mask = remove_small_objects(mask, self.min_size_px)
        return label(mask).astype(np.int32)
'''

README = """This directory was created by `pyclm new --demo`. It runs on the virtual
microscope; no microscope, Micro-Manager or downloaded data is needed.

  bar.toml           open loop: a bar of light sweeping across the field
  cells.toml         closed loop: threshold segmentation, then light the
                     outward-facing half of every cell (move_out)
  demo_methods.py    the threshold segmentation: a custom method, listed under
                     methods in pyclm_config.toml
  bar.tif, cells.tif synthetic cells the virtual microscope shows
  dry_run.yml        which TIF each position shows, and their pixel size
  schedule.toml      ten timepoints, three seconds apart
  pyclm_config.toml  the configuration (a calibration covering the whole camera)

Try
  pyclm check <this directory>
  pyclm preview <this directory> cells --image cells.tif
  pyclm run <this directory> --dry          (add --gui to watch, with the gui extra)

Then open the results: <experiment>.00.zarr in napari or Fiji, or in Python
  import pyclm.io; exp = pyclm.io.open("cells.00.zarr")
To run again, delete the outputs (*.zarr, *_imaging.tif, frames.*, events.*,
status.json, log.log).
"""


def synthetic_cells(
    seed: int, frames: int = FRAMES, size: int = SIZE, n_cells: int = 60
) -> np.ndarray:
    """
    A (frames, size, size) uint16 movie of round cells on a noisy background,
    drifting slowly outward from the centre (what move_out would encourage).
    """
    rng = np.random.default_rng(seed)
    # cells on a jittered grid, so that they rarely touch
    side = int(np.ceil(np.sqrt(n_cells)))
    pitch = size / side
    gy, gx = np.mgrid[0:side, 0:side]
    centres = (np.stack([gy.ravel(), gx.ravel()], axis=1) + 0.5) * pitch
    centres = centres[:n_cells] + rng.uniform(-0.2, 0.2, (n_cells, 2)) * pitch
    radii = rng.uniform(0.18, 0.3, n_cells) * pitch
    brightness = rng.uniform(1500, 4000, n_cells)
    outward = centres - size / 2
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1.0)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    movie = np.empty((frames, size, size), np.uint16)
    for t in range(frames):
        frame = np.full((size, size), 300.0, np.float32)
        here = centres + outward * 0.8 * t
        for (cy, cx), r, b in zip(here, radii, brightness, strict=True):
            y0, y1 = max(int(cy - 2 * r), 0), min(int(cy + 2 * r) + 1, size)
            x0, x1 = max(int(cx - 2 * r), 0), min(int(cx + 2 * r) + 1, size)
            d2 = (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2
            # a soft-edged disc
            frame[y0:y1, x0:x1] += b / (1.0 + np.exp((np.sqrt(d2) - r) / 1.5))
        frame += rng.normal(0, 60, frame.shape)
        movie[t] = np.clip(frame, 0, 65535).astype(np.uint16)
    return movie


def _swap(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"the demo expected {old!r} in the template")
    return text.replace(old, new, 1)


def create_demo(directory: Path) -> list[Path]:
    """Write the demo into ``directory`` (created if needed); existing files are kept."""
    import tifffile

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    binned = (
        "binning = 1\n",
        "binning = 4                   # the demo TIFs are 4x binned frames\n",
    )
    bar = _swap(OPEN_LOOP.format(name="bar"), *binned)
    bar = _swap(
        bar,
        "every_t = 5                   # image every 5th timepoint",
        "every_t = 2                   # image every 2nd timepoint",
    )
    bar = _swap(
        bar,
        "bar_speed = 1.0               # um per minute",
        "bar_speed = 600.0             # um per minute: fast, for a 30 s demo",
    )
    cells = _swap(CLOSED_LOOP.format(name="cells"), *binned)
    cells = _swap(
        cells,
        "every_t = 5                   # image every 5th timepoint",
        "every_t = 1                   # image every timepoint",
    )
    cells = _swap(
        cells,
        'method = "cellpose"\nmodel = "cpsam"',
        'method = "threshold"            # from demo_methods.py\nmin_size_px = 20',
    )
    files = {
        "bar.toml": bar,
        "cells.toml": cells,
        "schedule.toml": SCHEDULE,
        "pyclm_config.toml": CONFIG,
        "dry_run.yml": DRY_RUN,
        "demo_methods.py": METHODS,
        "README.txt": README,
    }
    written = []
    for filename, text in files.items():
        path = directory / filename
        if path.exists():
            continue
        path.write_text(text, encoding="utf-8")
        written.append(path)
    for filename, seed in (("bar.tif", 1), ("cells.tif", 2)):
        path = directory / filename
        if path.exists():
            continue
        tifffile.imwrite(path, synthetic_cells(seed), metadata={"axes": "TYX"})
        written.append(path)
    return written
