---
name: pyclm-microscope-bringup
description: Get PyCLM running on a microscope it has never run on - install, the Micro-Manager device interface, the .cfg presets, pyclm_config.toml (DMD/SLM affine and shape, camera ROI, pixel size, position mover and focus lock), and the first real run. Use when installing PyCLM on a new scope, when `pyclm check` or a first run fails on hardware, or for device-interface, pixel-size, affine or focus-lock problems.
---

# Bring PyCLM up on a microscope

Work in order. Each step ends with a check that must pass before the
next one. Never edit the microscope's Micro-Manager `.cfg` yourself; tell
the researcher what to change in Micro-Manager and why.

**Where the documentation is.** In a clone of the repository, the files
named below are under `documentation/` and `docs/`. After a plain pip
install, read the same pages on https://pyclm.readthedocs.io (for
example `documentation/first_time_setup.md` is
https://pyclm.readthedocs.io/en/latest/first_time_setup.html) and the
`docs/` notes on https://github.com/Harrison-Oatman/PyCLM/tree/main/docs.

## 0. Take an inventory of the rig

Ask the researcher, and write the answers down in a short `RIG.md` next
to `pyclm_config.toml` so later sessions do not ask again:

- the microscope body, camera, and objectives that will be used;
- the light-patterning device (DMD or other SLM) and the name of its
  Micro-Manager device adapter;
- the light source that feeds it, and how light is routed to it (a
  preset, a shutter, a filter turret position);
- the hardware focus lock, if any (Nikon Perfect Focus, Zeiss Definite
  Focus, Olympus ZDC, Leica AFC, none);
- the Micro-Manager version and install folder, and the `.cfg` in use;
- whether the Projector plugin calibration has been done for each
  objective.

Then show PyCLM working with no hardware, so a later failure is clearly
about the rig:

```bash
pyclm new demo --demo
pyclm run demo --dry --gui
```

## 1. Install and match the device interface

```bash
pip install "closed-loop-microscopy[gui]"      # add [cellpose] for GPU segmentation
mmcore use "<the Micro-Manager install folder>"   # tell pymmcore-plus which install to use
pyclm check <any experiment dir>                # pyclm new <dir> makes one
```

`mmcore` is pymmcore-plus's command (https://pymmcore-plus.github.io/pymmcore-plus/);
`mmcore install` downloads a Micro-Manager nightly if the PC has none, and
`mmcore list` shows what is installed. PyCLM does not need the
Micro-Manager GUI to run, but the GUI is needed for the `.cfg`, the
pixel-size calibration and the Projector calibration below.

pymmcore loads the device adapters of the Micro-Manager installation,
and both must be built for the same **device interface**. The interface
is the second-to-last number of the pymmcore version (`11.2.1.71.0` is
interface 71). `pyclm check` reports pymmcore's interface and the folder
whose adapters must match it.

- A pip install gets the newest pymmcore. That matches a recent
  Micro-Manager nightly, which is the usual case.
- If the adapters are older, install the pymmcore built for their
  interface together with a pymmcore-plus that supports it. Vendor DLLs
  that are not rebuilt often are the common cause. For example, the
  Mightex Polygon DLL on the developers' scope is interface 71 and needs
  `pip install "pymmcore==11.2.1.71.0" "pymmcore-plus==0.14.0"`.
- A clone run with `uv sync` is locked to that interface-71 pair (in
  `pyproject.toml`, `[tool.uv] constraint-dependencies`). On a rig with
  newer adapters, use a pip install, or change that constraint.
  `docs/stage5b-interactivity-design.md`, "Device interface", records why
  the lock exists.

A mismatch shows up as devices failing to load, not as a Python error.

**Check:** the `.cfg` loads in `pyclm check` with no device-interface
complaint.

## 2. The Micro-Manager configuration

PyCLM switches channels with config-group presets. The researcher needs:

- one group whose presets are the imaging channels;
- a preset that routes light to the patterning device;
- pixel-size presets for every objective (Devices → Pixel Size
  Calibration, or the Pixel Calibrator plugin), each conditioned on the
  objective's device property, and each experiment TOML naming the
  objective in `[config_groups]`. See
  `documentation/first_time_setup.md`, §3b: every µm in a pattern
  method comes from this pixel size.

`pyclm check` validates every preset the TOMLs name against the `.cfg`
and reports which pixel-size preset applies. If it reports "no pixel-size
preset matches", read its explanation. Either the objective preset and the
pixel-size condition disagree, which is fixed in Micro-Manager's
pixel-size calibration, or the condition is on a property no group sets,
which is only information.

**Check:** `pyclm check <dir> --mm-config <the .cfg>` reports no unknown
presets and the expected pixel size.

## 3. `pyclm_config.toml`

Every field is documented in `documentation/first_time_setup.md`, §3.

- `affine_transform` (camera to DMD, 2×3) and `slm_shape_h`/`slm_shape_w`
  come from the Projector plugin's calibration on the **unbinned, full**
  camera frame, per objective. Never type in numbers from another rig.
  1. In the Micro-Manager GUI, open Plugins → Device Control → Projector,
     with a uniform fluorescent sample (a highlighter-coated slide, or
     dye) in focus, and run Calibrate
     (https://micro-manager.org/Projector).
  2. With the GUI still open, run `scripts/calibration.ipynb` from a
     clone of the repository (needs `pip install
     "closed-loop-microscopy[calibration]"`, which brings pycromanager,
     and Micro-Manager's Tools → Options → "Run server on port 4827"
     on). It reads the Projector's mapping and prints the 2×3 affine
     (the median over the plugin's local maps); `core.get_slm_height`
     and `get_slm_width` in the same notebook give the DMD shape.
  3. Paste them into `pyclm_config.toml`, and check them with first light
     (§4), not by eye on the numbers.
  Keep one config per objective, as `pyclm_config.<objective>.toml`; only
  one may sit in an experiment directory.
- `camera_roi`: `pyclm check` prints the DMD footprint and suggests it.
- `position_mover`:
  - `"basic"`: XY then Z, no focus lock. Start here on any rig.
  - `"pfs"`: Nikon Perfect Focus only.
  - `"module:Class"`: any other focus lock. Write a `PositionMover`
    subclass as in `documentation/first_time_setup.md`, §4, and test it
    on one position before a long run.
- `focus_device`: the Micro-Manager Z device, if it is not `ZDrive`.
- `[output]`: `format` (`ome-zarr`, or `hdf5`), `pattern_policy`,
  `export_imagej`.

**Check:** `pyclm check` reads the config with no errors and prints the
mover it resolved to.

## 4. First light

1. `pyclm preview <dir> <condition> --snap`. Real frames come back with
   the presets applied, at the ROI's size.
2. A `full_on` experiment with one position and `steps = 3`. The
   stimulation frame must be lit edge to edge inside the ROI. If the
   light is shifted, rotated or mirrored, the affine is wrong.
   Recalibrate; do not compensate in a pattern method.
3. A `bar` experiment. The bar must move in +y for positive `speed_um_per_min`
   in the viewer (`pyclm run <dir> --gui`).

## 5. Focus

With `position_mover = "basic"`, check that the sample stays in focus
over a few positions and an hour, before trusting a long run.

With `position_mover = "pfs"`:

- `log.log` has one "move+focus took" line per position.
- If the PFS switches itself off, the mover turns focus maintenance off
  and on again and logs a warning.
- A `TimeoutError` lists the PFS status strings it saw. The lock value is
  `PFS_LOCKED_VALUE` in `src/pyclm/core/position_mover.py`; a subclass can
  change it if a different Ti2 firmware reports another string. Report the
  strings instead of guessing what they mean.

With a custom mover, log the time each move and focus takes, as the PFS
mover does, so slow positions show up in `log.log`.

## 6. The release checks

`tests/wet_tests/` in a clone holds the scripted hardware tests. See its
README, then `uv run wet list` there. Run `01_environment` to `03_hdf5`
on a new scope before trusting a long experiment. Item 02 expects a focus
lock per position; on a rig without one, set `position_mover = "basic"`
in its `config_template.toml` and say so in the report.

When something fails, collect `log.log`, `status.json` and the
`pyclm check` output before proposing a fix. Read `docs/known-issues.md`,
which may already describe it. If it is a PyCLM bug, help the researcher
open an issue at https://github.com/Harrison-Oatman/PyCLM/issues with
those files and the rig inventory.
