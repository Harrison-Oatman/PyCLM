# Testing guide: Stages 3 to 5b on the virtual and the real microscope

*For Harrison, 2026-09-09. Everything below "Dry" runs on any PC with the
repository checked out; everything below "At the microscope" needs the
scope. Do the dry part first; it takes about twenty minutes.*

## 0. Setup

```bash
git pull                     # branch core-refactor, commit after Stage 5b
uv sync --group test
uv run --group test pytest   # expect 233 passed, ~2 min
```

The dependency upgrade in this stage is the one thing that can bite:
pymmcore-plus 0.13.7 → 0.18.1 and pymmcore 11.2 → 12.5, which speaks
**Micro-Manager device interface 75** (the old one spoke 71). On the
microscope PC the installed Micro-Manager's device adapters must match, or
the configuration will not load (see §2.1 for how to tell and what to do).
The suite cannot check this; it is the first thing to test at the scope.

## 1. Dry

Make a scratch copy of a real experiment directory, or build one:

```bash
uv run pyclm new scratch_dry --template open-loop --name bar
# copy two TIFs into scratch_dry/ (any 2D or stack TIF; the test ones are in
# tests/dry_run_resources/tifs) and write scratch_dry/dry_run.yml:
#   positions:
#     - name: bar.00
#       source: mdck_fast_bar.tif
#     - name: bar.01
#       source: mdck_slow_bar.tif
```

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
uv run pyclm preview scratch_dry bar --image scratch_dry/mdck_fast_bar.tif --pixel-size-um 0.33
```

Open `scratch_dry/preview/bar.preview/pattern_overlay.png`: the bar over
the image. `preview.json` has the timings and the fraction lit.

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
(`t n/N | at bar.00 ...`) updating every second and the minimap with the two
positions, the current one filled. Then, in a third terminal or a Python
prompt:

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

### 1.4 The control window on the virtual microscope

```bash
uv run pyclm control scratch_dry --dry
```

- **Run tab**: Check → report. Rehearse (dry) → output scrolls, the buttons
  flip to Pause / Resume / Stop, the minimap and status line update. Try
  Pause, Resume, Set exposure (80 ms on 545), Set pattern parameter
  (`bar_speed` = `0.5`), Stop run. Each action appears in the output log as
  a written command file; the events table shows them afterwards.
- **Positions tab**: Load list is empty (a dry directory has none). Add
  current position (a row `bar.00` appears at the virtual stage's
  coordinates); change the experiment dropdown; Add again (`bar.01`);
  Preview here (a preview is written and summarised in the message line);
  Save → `PositionList.pos` appears; `pyclm check scratch_dry` now lists the
  positions.
- **Files tab**: open `bar.toml`, change the imaging exposure, add a
  `period` argument to the pattern, Validate, Save. Open the file in an
  editor: the value changed, the comments of the other tables are intact.
  Set an exposure to 0 and Validate: the schema's message appears.

### 1.5 The unit tests behind all of this

`tests/test_commands.py`, `tests/test_gui_widgets.py`, `tests/test_control.py`
(the window runs offscreen), `tests/test_schema.py`, `tests/test_check.py`,
`tests/test_preview.py`. `pytest -q tests/test_control.py` is the quick one.

## 2. At the microscope

Do these in order; each depends on the previous.

### 2.1 The core loads (the upgrade)

```bash
uv run python -c "import pymmcore; print(pymmcore.__version__)"
uv run python -c "from pymmcore_plus import CMMCorePlus; c = CMMCorePlus(); c.loadSystemConfiguration(r'C:\Program Files\Micro-Manager-2.0\<your>.cfg'); print(c.getLoadedDevices())"
```

If the second line fails with a *device interface version* error, the
installed Micro-Manager is older than pymmcore 12.5 expects. Two ways out:
install a current Micro-Manager 2.0 nightly next to the old one and point
`config_path` at a copy of the configuration there (vendor SDKs the old
install relied on, such as Nikon's, must be present for that install too),
or run `uv run mmcore install` to fetch matching device adapters into your
user directory. If neither is possible on that PC on the day, roll the
pins back (`git checkout a8e7df7 -- pyproject.toml uv.lock` is not enough:
edit `pyproject.toml` to `pymmcore-plus==0.13.7`, `pymmcore==11.2.1.71.0`,
remove `pymmcore-widgets`, then `uv sync`), and everything except the
positions tab's stage widgets keeps working.

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

### 2.5 The control window

```bash
uv run pyclm control <dir>
```

- **Positions**: the stage widgets move the stage and snap; Add current
  position after moving; Preview here; Save. Start MicroManager afterwards
  (or `pyclm check`) to confirm the saved list is read correctly, including
  the PFS offset if your stage reports one.
- **Run**: Start run (the window releases the microscope first; if it does
  not, the run's log will say the device is in use). Pause, Resume, Stop.
  When the run ends, the Positions tab should be live again.

### 2.6 What to report back

For each of 2.1 to 2.5: worked / did not, and for failures the traceback
or the line from `log.log`. For 2.3, whether the images match the
previous run. Anything that surprised you in the window's behaviour, even
if it worked.
