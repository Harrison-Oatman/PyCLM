# First-Time Setup Guide

This guide walks through everything required to run a PyCLM experiment on a new microscope system. By the end you will have:

- A working Python environment with PyCLM installed
- A hardware configuration file (`pyclm_config.toml`)
- A `PositionMover` matched to your focus-maintenance hardware
- A pair of experiment configuration files (`[experiment].toml` and `schedule.toml`)
- A position list exported from MicroManager

---

## 1. Install PyCLM

Create a virtual environment using [uv](https://docs.astral.sh/uv/) and install pyclm:

```bash
git clone https://github.com/Harrison-Oatman/PyCLM.git
cd PyCLM
uv sync --group dev
```

To use CellposeSAM segmentation, install the optional cellpose extras (requires a CUDA-capable GPU):

```bash
uv sync --extra cellpose
```

---

## 2. Configure MicroManager

PyCLM drives hardware through [pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/), which loads a standard MicroManager `.cfg` file.

**See [MicroManager documentation](https://micro-manager.org/Micro-Manager_Configuration_Guide)**

**Before running PyCLM you must have:**

1. A working MicroManager installation with device adapters for your camera, stage, and (if applicable) SLM.
2. A `.cfg` file that loads without errors in MicroManager. Run MicroManager once and use the Hardware Configuration Wizard to create and verify this file.
3. **Config groups** for any device state that changes between imaging channels (e.g. filter wheels, laser lines). PyCLM switches channels by calling `setConfig(group, preset)`, so each illumination condition you intend to image needs a named preset in a named group.

PyCLM does not use MicroManager's graphical interface at runtime — the `.cfg` file is the only dependency.

---

## 3. Create `pyclm_config.toml`

Place a `pyclm_config.toml` in the experiment directory or at the repository root. PyCLM searches the experiment directory first, then the working directory.

```toml
# Absolute path to your MicroManager .cfg file
config_path = "C:/Program Files/Micro-Manager-2.0/MyScope.cfg"

# SLM (DMD) pixel dimensions [height, width]
slm_shape_h = 1140
slm_shape_w = 912

# Affine transform from camera pixel coordinates to SLM pixel coordinates.
# This is a 2x3 matrix [[a, b, tx], [c, d, ty]].
# Obtain it by running the MicroManager Projector plugin calibration.
affine_transform = [[-0.29, -0.002, 939.9], [0.004, -0.579, 1505.2]]

# Optional. MicroManager focus (Z) device selected at startup. Default "ZDrive".
focus_device = "ZDrive"

# Optional. Seconds to wait after the hardware reports ready, before each snap.
# Default 1.0. Lower it on a fast, stable stage; it is paid once per acquisition.
settle_time_seconds = 1.0

# Optional. Camera ROI [x, y, width, height] in unbinned pixels, set when a run
# (or `pyclm preview --snap`) starts. The DMD usually lights only part of the
# field: `pyclm check` prints the region it covers and suggests it here, so
# that every pixel imaged can be stimulated. Omit to leave the camera as it is.
# The affine above stays calibrated in the full frame; PyCLM composes the offset.
camera_roi = [256, 128, 1536, 1792]

# Optional. How data is stored and exported.
[output]
format = "ome-zarr"          # "ome-zarr" (default) or "hdf5" (PyCLM's original layout)
pattern_policy = "on_change" # ome-zarr only: store each distinct pattern once ("on_change"),
                             # only at timepoints with a saved imaging frame ("imaging"),
                             # or not at all ("none"); stored in camera and DMD coordinates
export_imagej = true         # write ImageJ hyperstacks next to the data when the run finishes
```

By default (`format = "ome-zarr"`) each experiment is written as an
[OME-Zarr](https://ngff.openmicroscopy.org/) store (`<experiment>.zarr/`) that
Fiji, napari, QuPath and plain Python (`zarr`, `dask`) open directly, with one
image per acquisition cadence so channels imaged every N timepoints never show
blank frames, segmentation as NGFF labels, DMD patterns stored once per
distinct pattern, and a `frames.parquet`/`frames.csv` table of every frame and
stimulation event. Whichever format is used, `pyclm.io.open(path)` reads it
and `convert_hdf5s` (or the automatic export) produces ImageJ stacks. See
[Data format and export](data_format.md) for the full description.

If your microscope has no SLM, set the shape to the physical DMD resolution anyway — PyCLM will skip hardware calls when no SLM device is detected.

---

## 4. Choose or Implement a PositionMover

PyCLM needs to know how to move to an imaging position on your hardware. Three choices are available:

| Class | When to use |
|---|---|
| `BasicPositionMover` | Default. Simple XY + Z move, no focus maintenance. |
| `PFSPositionMover` | Nikon Ti2 with Perfect Focus System. |
| Custom subclass | Any other hardware-autofocus system. |

### Using a built-in mover

Pass a mover instance to `run_pyclm` or `Controller`:

```python
from pyclm import run_pyclm
from pyclm import PFSPositionMover   # Nikon PFS
# from pyclm import BasicPositionMover  # simple XYZ

run_pyclm(
    "path/to/experiment_dir",
    position_mover=PFSPositionMover(),
)
```

### Writing a custom mover

Subclass `PositionMover` and implement `move_to`. The method receives a `MicroscopePosition` (with attributes `x`, `y`, `z`, and an `extras` dict for optional device values) and a `core` object that exposes the MicroManager API.

```python
from pyclm import PositionMover, run_pyclm

class MyFocusMover(PositionMover):
    def move_to(self, position, core) -> tuple[bool, float]:
        # 1. Move XY
        core.setXYPosition(position.x, position.y)

        # 2. Move Z
        core.setPosition(position.z)

        # 3. Apply any optional hardware-specific values stored in extras.
        #    For example, a laser autofocus offset:
        af_offset = position.extras.get("AFOffset")
        if af_offset is not None:
            core.setAutoFocusOffset(af_offset)
            # ... poll or wait for hardware confirmation ...

        # Return (z_was_adjusted: bool, actual_z: float)
        return True, core.getZPosition()

run_pyclm("path/to/experiment_dir", position_mover=MyFocusMover())
```

The `extras` dict is populated from any devices in the position list beyond the XY and Z stages (see Section 5 below).

---

## 5. Export a Position List from MicroManager

Open MicroManager and navigate to **Devices → Stage Position List**. Add all imaging positions, assigning names that match your experiment TOML files (see Section 6). Save the list as `PositionList.pos` and place it in the experiment directory.

**Position naming convention:** Each position name must start with the stem of a `.toml` filename in the same directory, followed by `-` and any suffix. For example:

```
feedback_ctrl-pos1
feedback_ctrl-pos2
open_loop-pos1
```

These three positions would use `feedback_ctrl.toml` for the first two and `open_loop.toml` for the third.

PyCLM also supports the Nikon Elements `multipoints.xml` format (exported from the xy-positions tab of an NDAcquire). If both files are present, `PositionList.pos` takes precedence.

### Grids

A position can be a **grid of tiles** imaged as one stitched frame, so a
pattern method sees, and lights, a region larger than one field. Grids are
made in MicroManager with the Stage Position List's **Create Grid** (the
tile creator) and need nothing in the experiment TOML:

1. Once per objective, run MicroManager's **Pixel Calibrator** so that the
   pixel-size affine knows the camera's orientation; the tile creator lays
   tiles out with it, and PyCLM stitches by tile row and column, so tiles
   abut only if this is right.
2. Set the camera ROI you will run with (`camera_roi` in `pyclm_config.toml`;
   `pyclm check` prints the region the DMD covers). Create Grid spaces tiles
   by the *current* image size, so the ROI must be active when the grid is
   created. Use overlap 0 for stimulated grids: the overlap strip would be
   lit from two tiles.
3. In Create Grid, set the **prefix** to the experiment's TOML stem
   (`tissue` for `tissue.toml`), mark the corners, and create. MicroManager
   writes one entry per tile, labelled `tissue-1-000_000`,
   `tissue-1-001_000`, … with `GridRow` / `GridCol`, each with its own z
   and, on a Nikon, PFS offset interpolated between the corners.
4. Save the list as `PositionList.pos` in the experiment directory.

PyCLM folds those entries into one position, `tissue.1`, at the centre of
the tiles. `pyclm check` reports each grid (rows × columns, spacing, and,
given `camera_roi` and `--pixel-size-um`, the overlap in pixels). At each
timepoint every channel visits all the tiles (the stimulation frame with
its tile's own DMD image) and publishes one stitched frame; the pattern
method's `pattern_shape` is the stitched frame and the pattern it returns
is cut into one DMD image per tile. Grids need the OME-Zarr format. Do not
delete tiles from a grid; PyCLM refuses an incomplete rectangle.

---

## 6. Write Experiment TOML Files

`uv run pyclm new path/to/experiment_dir --name my_experiment` writes a
starting set of files (an open-loop or a closed-loop template) with comments
saying what to change; the rest of this section explains them.

Each experiment type is described by a TOML file. Multiple positions can share the same experiment file; one experiment file can therefore run simultaneously at several locations.

Below is a fully annotated example for a feedback-controlled optogenetic experiment:

```toml
# ── Optional timing offsets (in timepoints, not seconds). Keep them above the ─
# ── first table: a key written after a table header belongs to that table. ──
# t_delay = 5    # wait N timepoints before starting this experiment
# t_stop  = 100  # stop after N timepoints (0 = run until schedule ends)

# ── Optional: device state applied to every channel in this experiment ──────
[config_groups]
# "GroupName" = "PresetName"  (MicroManager config group)
Shutter = "Open"

[device_properties]
# "DeviceName-PropertyName" = value
# "Laser488-PowerSetpoint" = 5.0


# ── Imaging defaults (all channels inherit these unless overridden) ──────────
[imaging]
exposure = 50       # exposure time in milliseconds
every_t = 1         # acquire every N timepoints (1 = every timepoint)
save = true         # write to HDF5
binning = 1         # camera binning (1, 2, or 4)

[imaging.config_groups]
# Config groups applied specifically to all imaging channels
LightPath = "Confocal"


# ── Channel definitions ──────────────────────────────────────────────────────
# "group" is the MicroManager config group used to switch channels.
# "presets" lists the presets within that group that will be imaged.
[channels]
group = "FP"
presets = ["GFP", "RFP"]

# Override imaging defaults for specific channels:
[channels.GFP]
exposure = 100
every_t = 1

[channels.RFP]
exposure = 50
every_t = 2   # image RFP half as often as GFP


# ── Stimulation channel ──────────────────────────────────────────────────────
# This is the light delivery channel — the DMD pattern is applied here.
[stimulation]
exposure = 200      # ms; set to 0 to deliver no stimulation
every_t = 1

[stimulation.config_groups]
LightPath = "DMD"

[stimulation.device_properties]
# "Sola-PowerSetpoint" = 20.0


# ── Segmentation (optional) ──────────────────────────────────────────────────
# Remove this section (or set method = "none") for open-loop experiments.
[segmentation]
method = "cellpose"
# Additional kwargs are forwarded to the segmentation method constructor:
# model = "cpsam"  # cellpose built-in model
# model = "finetuned_mcf10a"  # custom pre-trained model


# ── Pattern generation ───────────────────────────────────────────────────────
# "method" must match a registered PatternMethod name 
# either built into pyclm (see documentation/method_zoo.md),
# or custom (see Section 10 below).
# All other keys are forwarded as constructor kwargs to the pattern method.
[pattern]
method = "circle"
rad = 150            # circle radius in µm

```

---

## 7. Write `schedule.toml`

Place one `schedule.toml` in the experiment directory. It controls the overall timing of the multi-experiment:

```toml
[timing]
steps = 120                  # total number of timepoints
interval_seconds = 30.0      # time between consecutive timepoints
setup_time_seconds = 2.0     # delay before the first timepoint
time_between_positions = 2.0 # pause between consecutive positions within a timepoint
```

---

## 8. Experiment Directory Layout

Before running, your experiment directory should contain:

```
experiment_dir/
├── PositionList.pos       # position list from MicroManager (preferred)
│   or multipoints.xml     # legacy alternative
├── schedule.toml
├── feedback_ctrl.toml     # one .toml per experiment type
├── open_loop.toml
└── pyclm_config.toml      # optional; falls back to repository root
```

PyCLM writes output files alongside the configuration files:

```
experiment_dir/
├── feedback_ctrl.pos1.hdf5        # or feedback_ctrl.pos1.zarr/ with format = "ome-zarr"
├── feedback_ctrl.pos2.hdf5
├── open_loop.pos1.hdf5
├── feedback_ctrl.pos1_imaging.tif # ImageJ hyperstacks, exported when the run ends
├── frames.parquet / frames.csv    # (ome-zarr) one row per frame and stimulation event
├── plan.useq.yaml                 # the acquisition plan PyCLM derived from your files
└── log.log
```

`plan.useq.yaml` is a [useq-schema](https://pymmcore-plus.github.io/useq-schema/)
`MDASequence` describing every position, channel, exposure and the timing,
plus PyCLM's own settings under `metadata.pyclm`. It is also stored inside
each HDF5 file. Keep it with the data: it is the exact record of what was
scheduled.

---

## 9. Check, Preview, Run

Before the microscope is touched:

```bash
uv run pyclm check path/to/experiment_dir
uv run pyclm preview path/to/experiment_dir my_experiment --image a_snapped_frame.tif
```

`check` reports every problem in the files, with the file and key it
concerns; `preview` runs the segmentation and the pattern method once on an
image and writes what they produce. `pyclm run` performs the same check and
refuses to start on errors. See [The pyclm command](command_line.md).

**From the command line:**

```bash
uv run pyclm run path/to/experiment_dir
```

Pass `--config` if `pyclm_config.toml` is not in the experiment directory or repository root:

```bash
uv run pyclm run path/to/experiment_dir --config path/to/pyclm_config.toml
```

Use `--dry` to run a full rehearsal without connecting to the microscope:

```bash
uv run pyclm run path/to/experiment_dir --dry
```

In dry mode, PyCLM reads simulated images from TIF files placed inside the experiment directory. Positions are loaded from `PositionList.pos` or `multipoints.xml` if present, and each TIF is matched to a position by label (e.g. `on.00.tif` matches position `on.00`) or stem (e.g. `on.tif` matches any position with stem `on`). For explicit control, add a `dry_run.yml` to the experiment directory mapping position names to TIF files:

```yaml
pixel_size_um: 1.333   # the size of one pixel of the TIFs (default 0.33)
binning: 4             # how the TIFs were binned, relative to the camera the
                       # affine transform was calibrated for (default 1)
positions:
  - name: on.00
    x: 0.0
    y: 0.0
    source: on.tif
  - name: off.00
    x: 500.0
    y: 0.0
    source: off.tif
```

The two top-level keys are optional and may also stand alone in a
`dry_run.yml` without `positions`, in which case the positions come from the
position list or the TIF names as above. `pixel_size_um` is what the pattern
methods and the output metadata see; `binning` scales the camera-to-DMD
affine of `pyclm_config.toml`, which was calibrated on the unbinned camera.
Without it, images binned 4x land mostly off the DMD and the stored DMD
pattern is nearly empty even though the pattern in camera coordinates is
fine. `pyclm preview --image` uses the same two keys when
`--pixel-size-um` is not given.

Add `--gui` to open a live Napari viewer that updates as data is written:

```bash
uv run pyclm run path/to/experiment_dir --dry --gui
```

`--gui` can also be used during a real experiment to monitor output in real time.

**Programmatically** (required when using a custom `PositionMover` or custom pattern/segmentation methods):

```python
from pyclm import run_pyclm, PFSPositionMover

run_pyclm(
    "path/to/experiment_dir",
    position_mover=PFSPositionMover(),
    # segmentation_methods={"my_seg": MySegmentationMethod},
    # pattern_methods={"my_pattern": MyPatternMethod},
)
```

The experiment can be aborted at any time with `Ctrl+C`. Data already written to HDF5 is not lost.

---

## 10. Custom Pattern and Segmentation Methods

For a more detailed explanation, see [Writing custom pattern methods](custom_pattern_methods.md).

### Pattern method

Subclass `PatternMethod` and implement `generate`. Call `add_requirement` in `__init__` to declare what image data the method needs at each timepoint.

```python
import numpy as np
from pyclm import PatternMethod

class MyPattern(PatternMethod):
    name = "my_pattern"  # used as method = "my_pattern" in the TOML

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        # Request the raw GFP image and its segmentation at every timepoint:
        self.add_requirement("GFP", raw=True, seg=True)

    def generate(self, context) -> np.ndarray:
        raw = context.raw("GFP")         # np.ndarray, camera coordinates
        seg = context.segmentation("GFP")

        # Return a float32 array in [0, 1] with the same shape as the camera ROI.
        pattern = (seg > 0).astype(np.float32)
        return pattern
```

Register and run:

```python
run_pyclm("path/to/experiment_dir", pattern_methods={"my_pattern": MyPattern})
```

### Segmentation method

Subclass `SegmentationMethod` and implement `segment`:

```python
import numpy as np
from pyclm import SegmentationMethod

class MySegmentation(SegmentationMethod):
    name = "my_seg"

    def __init__(self, experiment_name, threshold=128, **kwargs):
        super().__init__(experiment_name)
        self.threshold = threshold

    def segment(self, data: np.ndarray) -> np.ndarray:
        # Return a label image (integer array, 0 = background).
        from skimage.measure import label
        binary = data > self.threshold
        return label(binary).astype(np.int32)
```

```python
run_pyclm("path/to/experiment_dir", segmentation_methods={"my_seg": MySegmentation})
```

The method name is then available as `method = "my_seg"` in the `[segmentation]` block of any experiment TOML.
