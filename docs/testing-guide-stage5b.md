# Testing guide: Stages 3 to 5b on the virtual and the real microscope

*For Harrison, 2026-09-09. Everything below "Dry" runs on any PC with the
repository checked out; everything below "At the microscope" needs the
scope. Do the dry part first; it takes about twenty minutes.*

## 0. Setup

```bash
git pull                     # branch core-refactor, commit after Stage 5b
uv sync --group test
uv run --group test pytest   # expect 232 passed, ~2 min
```

pymmcore must speak the same **Micro-Manager device interface** as the
installed device adapters. The Mightex Polygon adapter is built for
interface 71, so the lock now resolves to the last interface-71 stack
(pymmcore 11.2.1.71.0, pymmcore-plus 0.14.0); the
suite passes on it. `pyclm check` prints the interface pymmcore speaks.
§2.1 is what to run if a configuration still fails to load.

## 1. Dry

Make a scratch copy of a real experiment directory, or build one:

```bash
uv run pyclm new scratch_dry --template open-loop --name bar
# copy two TIFs into scratch_dry/ (any 2D or stack TIF; the test ones are in
# tests/dry_run_resources/tifs) and write scratch_dry/dry_run.yml:
#   pixel_size_um: 1.333      # the test TIFs are 4x binned frames of the
#   binning: 4                # camera the template's affine was calibrated on
#   positions:
#     - name: bar.00
#       source: mdck_fast_bar.tif
#     - name: bar.01
#       source: mdck_slow_bar.tif
```

Without the first two lines the calibrated affine puts the bar almost
entirely off the DMD (the DMD lit fraction in `preview.json` is 0
instead of about 0.2), which is what you saw.

### 1.1 The check

```bash
uv run pyclm check scratch_dry
```

Expect info lines and no errors (the config's `config_path` will be
reported as not found: that is an info line, not an error). Then break
something and check again: misspell `bar_speed` in `bar.toml`, or write
`t_delay = 3` at the end of the file after `[pattern]`. Each should be an
error naming the file and key, with a suggestion where there is one.

### 1.2 Preview

```bash
uv run pyclm preview scratch_dry bar --image scratch_dry/mdck_fast_bar.tif
```

Open `scratch_dry/preview/bar.preview/pattern_overlay.png`: the bar over
the image, and `pattern_dmd.tif`: the same bar on the DMD. `preview.json`
has the timings, the pixel size and binning it used, and the fractions lit
in camera and DMD space.

### 1.3 A rehearsal with commands

Terminal 1:

```bash
uv run pyclm run scratch_dry --dry
```

Terminal 2, while it runs:

```bash
uv run pyclm gui scratch_dry
```

Expect the napari window with a **Run** dock on the right: the status line
(`t n/N | at bar.00 ...`) updating every second, the minimap with the two
positions (the current one filled), and the positions list. The layer list
has one `imaging/545` layer and one `imaging/pattern` layer, not one per
position: the position slider under the canvas, the list, a click on the
minimap and the `[` / `]` keys switch between `bar.00` and `bar.01`. Tick
**Follow the run** and the viewer should jump to whichever position is
being acquired. Drag the contrast slider: it should stay where you put it
as new frames arrive (the **Auto-contrast** box unticks itself); **Normalise
now** resets it. Then, in a third terminal or a Python prompt:

```python
from pyclm.commands import write_command
write_command("scratch_dry", {"command": "pause"})      # the status line says "paused"; t stops advancing
write_command("scratch_dry", {"command": "resume"})     # it continues
write_command("scratch_dry", {"command": "set_exposure", "experiment": "bar.00", "channel": "545", "ms": 80})
write_command("scratch_dry", {"command": "set_pattern", "experiment": "bar.00", "parameters": {"bar_speed": 0.5}})
write_command("scratch_dry", {"command": "stop_run"})   # the run ends after the current timepoint
```

Afterwards, in Python:

```python
import pyclm.io
with pyclm.io.open("scratch_dry/bar.00.zarr") as exp:
    print(exp.events.to_pandas())      # command rows, the exposure row with source "command", the pattern row
    print(exp.frames.to_pandas()[["t", "channel", "exposure_ms"]])   # 80 ms on 545 frames after the change
```

and look at `scratch_dry/commands/done/` (the processed command files) and
`status.json` (`done: true`, `commands_applied`).

### 1.4 The unit tests behind all of this

`tests/test_commands.py`, `tests/test_gui_widgets.py`, `tests/test_viewer.py`,
`tests/test_schema.py`, `tests/test_check.py`, `tests/test_preview.py`.
`pytest -q tests/test_commands.py` is the quick one.

## 2. At the microscope

Do these in order; each depends on the previous.

### 2.1 The core loads (the upgrade)

```bash
uv run python -c "import pymmcore; print(pymmcore.__version__)"
uv run python -c "from pymmcore_plus import CMMCorePlus; c = CMMCorePlus(); c.loadSystemConfiguration(r'C:\Program Files\Micro-Manager-2.0\<your>.cfg'); print(c.getLoadedDevices())"
```

If the second line fails with a *device interface* error, the message now
names the interface pymmcore speaks (71 with the current lock). Every
adapter DLL in the Micro-Manager install must be built for that interface:
use the Micro-Manager install the Polygon adapter came with. When Mightex
ships an interface-75 adapter, delete the three `constraint-dependencies`
lines in `pyproject.toml`, run `uv lock` and `uv sync`, and repeat this
section on the interface-75 stack (the suite already passes on it).

### 2.2 A snap through the new stack

```bash
uv run pyclm preview <a real experiment dir> <experiment> --snap
```

The channel's presets and exposure are applied and one frame snapped per
required channel; the overlay under `preview/` should look like the
microscope's own snap. This exercises `RealMicroscopeCore` on the upgraded
pymmcore-plus.

### 2.3 A short known run

Take an experiment you have run before, set `steps` to about 10 with the
usual interval, and:

```bash
uv run pyclm check <dir>      # presets and devices are now checked against the .cfg
uv run pyclm run <dir> --gui
```

Compare the frames with the previous run of the same experiment (exposure,
intensity, the pattern on the DMD). The viewer's minimap should show the
real positions with the field of view drawn once the first frames arrive.

### 2.4 Commands during the run

While 2.3 runs, from another terminal, the same commands as in 1.3, in
particular a `set_property` on a laser and a `set_position` with a small
z change. Confirm in `events.parquet` that each was applied at the next
timepoint, and in the frames table that the new value is in force from
that timepoint on.

### 2.5 What to report back

For each of 2.1 to 2.4: worked / did not, and for failures the traceback
or the line from `log.log`. For 2.3, whether the images match the
previous run. Anything that surprised you in the window's behaviour, even
if it worked.
