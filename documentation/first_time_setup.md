# First-Time Setup Guide

This guide walks through everything required to run a PyCLM experiment on a new microscope system. By the end you will have:

- A working Python environment with PyCLM installed
- A hardware configuration file (`pyclm_config.toml`)
- A `PositionMover` matched to your focus-maintenance hardware
- A pair of experiment configuration files (`[experiment].toml` and `schedule.toml`)
- A position list exported from MicroManager

---

## 1. Install PyCLM

Clone the repository and create a virtual environment using [uv](https://docs.astral.sh/uv/):

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
```

If your microscope has no SLM, set the shape to the physical DMD resolution anyway — PyCLM will skip hardware calls when no SLM device is detected.

---

## 4. Implement a PositionMover

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

**Position naming convention:** Each position name must start with the stem of a `.toml` filename in the same directory, followed by `.` and any suffix. For example:

```
feedback_ctrl.pos1
feedback_ctrl.pos2
open_loop.pos1
```

These three positions would use `feedback_ctrl.toml` for the first two and `open_loop.toml` for the third.

PyCLM also supports the legacy `multipoints.xml` format (exported from older MicroManager versions via NIS-Elements). If both files are present, `PositionList.pos` takes precedence.

---

## 6. Write Experiment TOML Files

Each experiment type is described by a TOML file. Multiple positions can share the same experiment file; one experiment file can therefore run simultaneously at several locations.

Below is a fully annotated example for a feedback-controlled optogenetic experiment:

```toml
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


# ── Optional timing offsets (in timepoints, not seconds) ────────────────────
# t_delay = 5    # wait N timepoints before starting this experiment
# t_stop  = 100  # stop after N timepoints (0 = run until schedule ends)
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
├── feedback_ctrl.pos1.hdf5
├── feedback_ctrl.pos2.hdf5
├── open_loop.pos1.hdf5
└── log.log
```

---

## 9. Run the Experiment

**From the command line:**

```bash
uv run pyclm path/to/experiment_dir
```

Pass `--config` if `pyclm_config.toml` is not in the experiment directory or repository root:

```bash
uv run pyclm path/to/experiment_dir --config path/to/pyclm_config.toml
```

Use `--dry` to do a full rehearsal without connecting to the microscope (images are read from a `tif-source/` folder in the working directory):

```bash
uv run pyclm path/to/experiment_dir --dry
```

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
