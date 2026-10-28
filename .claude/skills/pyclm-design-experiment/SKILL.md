---
name: pyclm-design-experiment
description: Turn a researcher's optogenetic experiment idea into a PyCLM experiment directory that passes `pyclm check` and runs on the virtual microscope. Use when someone wants to set up, plan, or rehearse a PyCLM run, choose a pattern method, write experiment/schedule TOMLs, or asks "how do I run X with PyCLM".
---

# Design a PyCLM experiment

Goal: a directory the researcher can take to the microscope, with every
file checked and a dry run that shows the pattern they meant. Do not
write a custom pattern method until the built-in ones are ruled out; for
that, use the `pyclm-write-pattern-method` skill.

**The researcher decides.** You propose; the researcher approves. Before
writing files, and again before handing over, put each design decision to
them with a few sentences of plain prose that give the context that
matters for it, then wait for their answer. Decisions that change the
science need approval, for example:

- which conditions are tested (speeds, intensities, drugs);
- how many positions each condition gets, and which are controls;
- the final list of positions for a run;
- leaving a position or run out of an analysis.

Choices that do not change the science you make yourself and mention
briefly: file layout, names, comments, how a check is run. Keep each
approval short and about one decision. A long list of small approvals
trains people to approve without reading, and then the researcher has
lost control of the design without noticing.

**Where the documentation is.** In a clone of the repository, the files
named below are under `documentation/` and `src/`. After a plain pip
install, read the same pages on https://pyclm.readthedocs.io (for example
`documentation/method_zoo.md` is
https://pyclm.readthedocs.io/en/latest/method_zoo.html), and the source on
https://github.com/Harrison-Oatman/PyCLM, or in the installed package
(`python -c "import pyclm; print(pyclm.__file__)"`).

## 1. Ask what the biology needs, not what the files need

Get answers to these before writing anything. Ask only what is missing:

- **What light, where, when?** For example: a bar moving at 1 µm/min, the
  outward half of each cell, cells below a reporter threshold, or the
  whole field pulsed every 10 minutes. This picks the pattern method.
- **Does the pattern depend on the image?** If not, the run is open loop
  and needs no segmentation. If it does, the run is closed loop: which
  channel is segmented, and with what? The built-in segmentation is
  `method = "cellpose"` (CellposeSAM), which needs
  `pip install "closed-loop-microscopy[cellpose]"` and, in practice, an
  NVIDIA GPU: on a CPU one frame can take longer than the interval. A
  simple threshold or a lab's own model is written as a segmentation
  method (`pyclm-write-pattern-method`, last section).
- **How many conditions, and how many positions each?** Each condition is
  one experiment TOML. Each position names its TOML by its label.
  Suggest unstimulated control positions imaged the same way. PyCLM has
  no built-in "no light" method: a dark control is a condition whose
  pattern method returns zeros, a few lines written with
  `pyclm-write-pattern-method`.
- **Duration and cadence.** The interval between timepoints in seconds
  (`interval_seconds`), the number of timepoints (`steps`), and, counted
  in timepoints, how often to image (`[imaging] every_t`, or per channel)
  and how often to stimulate (`[stimulation] every_t`). For example,
  `interval_seconds = 60`, imaging `every_t = 5` and stimulation
  `every_t = 1` images every 5 min and updates the light every minute.
  All positions are visited within one interval, so the interval must be
  long enough for every position's channels (`pyclm check` estimates it).
- **Which microscope.** An existing `pyclm_config.toml` for that scope?
  If there is none and the scope has never run PyCLM, stop and use
  `pyclm-microscope-bringup` first. A dry run needs no microscope.
  Take channel and preset names from that scope's `.cfg`; every lab
  names them differently.

## 2. Pick the pattern method

Read the method zoo page (https://pyclm.readthedocs.io/en/latest/method_zoo.html,
or `method_zoo.html` of a local docs build). Map the description to a
built-in method and its arguments. The common ones:

| Want | `method` |
|---|---|
| A fixed shape or image | `circle`, `rectangle`, `full_on`, `image_mask` (a TIF) |
| Several doses in one field (quadrants, a spot array) | `dose_grid` |
| Field of moving stripes (negative speed reverses) | `bar`, `sawtooth`, `bar_bounce`, `spokes`, `rings` |
| Whole field (or still stripes) pulsed on a schedule | `global_pulse`, `pulse_schedule`, `blinking_bar`, `chirp` (a period that sweeps) |
| Any method pulsed on and off | `pulsed` with `[pattern.inner]` |
| Light whole cells: all, a random fraction, or chosen by area / intensity / position; a band at the colony edge | `all_cells` (`invert` for the background), `random_cells` (needs tracks), `select_cells` |
| Light one side of each cell (cells move towards the light) | `move_out`, `move_in`, `move_rotate`, `move_sequence` (a sequence of `u`, `d`, `l`, `r`, `s`) |
| Hold each nucleus at a target | `nucleus_clamp` (`controller = "bang_bang"`, `"p"`, `"pi"`), `target_image` (towards a target image), `relative_clamp` (a fraction of each cell's own range) |
| Hold the field's mean nuclear intensity with one input | `population_clamp` (a whole-field dose, the exposure, or a laser power) |
| Light the middle or the poles of an embryo | `embryo_inner`, `embryo_outer` |
| Different methods on two halves of the field | `split` with `[pattern.left]`/`[pattern.right]` or `top`/`bottom` |
| Keep the cells centered, undo drift, adjust an exposure, or step a setting during the run | `recenter`, `drift_correct`, `auto_exposure`, `settings_schedule`, each around an optional `[pattern.inner]` |
| Check the DMD calibration on a fluorescent slide | `game_of_life` |

`global_cycle`, `move_down` and `fb_bounce` are deprecated fixed cases of
`global_pulse` and `move_sequence`; `pyclm check` warns and gives the
equivalent arguments. Use the general method.

Arguments follow one convention: a quantity with a unit names it
(`radius_um`, `speed_um_per_min`, `step_min`; time is always in
minutes), fractions are 0–1, closed-loop methods need `channel`, and
every method takes `start_min` / `stop_min` (dark outside that window).
Methods and arguments renamed in 2.0 (`direction`, `rotate_ccw`,
`rotate_bar`, `binary_nucleus_clamp`, `centered_image`, `rad`,
`bar_speed`, `duty_cycle` + `period`, ...) still run with a warning;
`pyclm check --fix <dir>` rewrites the TOMLs.

Each method's card on the method zoo page has a **Keywords** table: every
argument the method takes, its default ("required" when it has none) and
what it does, with the generic `start_min` / `stop_min` in one table at
the top of the Methods section. Take argument names, units and defaults
from those tables, not from the source; they are built from the code
and its `:param` lines, and a test keeps them complete. The cards and
tables are generated when the docs build, so `documentation/method_zoo.md`
in a clone does not contain them; without network, build the docs
(`uv run --group docs sphinx-build -b html documentation <out>`). A method registered
by the lab (the `methods` key of `pyclm_config.toml`) is not in the zoo:
read its file. `pyclm check` rejects unknown
arguments, so never guess them.

## 3. Write the directory

```bash
pyclm new <dir> --template open-loop --name <condition>    # or closed-loop
```

Then edit:

- `<condition>.toml`: one per condition. `[channels]` names one
  Micro-Manager config group and the presets in it to image.
  `[stimulation]` is the light-patterning exposure: its config groups
  route light through the DMD, its `exposure` is how long the pattern is
  shown each time,
  `[pattern]` holds `method` plus its arguments, `[segmentation]` is only
  for closed loop (`channels = ["545", "638"]` gives the method one color
  image of several imaging channels; `auto_normalize = true` lets Cellpose
  scale each frame by its own percentiles instead of fixed
  `normlow`/`normhigh` counts). `[channels]` may be omitted for a stimulation-only run.
  Keys written after a table header belong to that table, so top-level
  keys (`t_delay`, `t_stop`) go above the first table.
- `schedule.toml`: `[timing]` `steps`, `interval_seconds`,
  `setup_time_seconds`, `time_between_positions`. Exactly one schedule
  per directory (`schedule.<label>.toml` is allowed).
- Positions: a Micro-Manager position list (Devices → Stage Position
  List, saved as `PositionList.pos`) or a Nikon Elements
  `multipoints.xml`, whose labels are `<condition>.<anything>`, for
  example `bar60.01`. PyCLM takes everything before the first `.` as the
  TOML's stem, so condition names cannot contain a `.`. Grids made with
  Create Grid are supported (`documentation/first_time_setup.md`, §5).
- `pyclm_config.toml`: copy the scope's working one. Do not invent
  `affine_transform` or `slm_shape_*`, and never copy them from another
  rig or from an example; they come from that scope's calibration. A dry
  run works with the template's values.

`documentation/experiment_tomls.md` has every key.

## 4. Check, preview, rehearse

```bash
pyclm check <dir>                      # add --mm-config file.cfg off the scope
pyclm preview <dir> <condition> --image frame.tif --pixel-size-um <um per pixel>
pyclm preview <dir> <condition> --image frame.tif --t 0:<steps-1>:<every>   # the whole schedule
pyclm run <dir> --dry --gui
```

- Fix every error `check` reports, and read its warnings aloud to the
  researcher. The timing-budget warning means the positions do not fit
  in the interval.
- `preview` writes the pattern for one image. Open it with the researcher
  and confirm it is what they meant before going to the scope.
- `preview --t START:STOP[:STEP]` runs the method over a stretch of the
  schedule in seconds (the end is included). Read `pattern_montage.png`
  yourself (timepoints and minutes are captioned on each panel) and check
  it against what the researcher described: the bar's direction and
  speed, a dark baseline from `t_delay`, pulses at the right times. Then
  show them the montage or `pattern_overlay.gif`. `preview.json` lists,
  per timepoint, whether light is delivered and how much of the field is
  lit. A closed loop sees one still image at every timepoint, so this
  checks its timing, not its response to moving cells; that needs the
  dry run.
- A dry run needs TIFs in the directory, named by position label or
  condition stem, or mapped in `dry_run.yml`. Set `pixel_size_um` and
  `binning` there if the TIFs are not full-resolution camera frames.
  Without them the DMD pattern lands off the field.
- The one-timepoint delay: a pattern generated from the data of
  timepoint t is shown from t+1 (settings a method changes follow the
  same rule; `documentation/custom_pattern_methods.md`, "Changing
  settings from a pattern method"). Explain this if the researcher counts
  timepoints.

## 5. Hand over

Tell the researcher:

- what each file sets;
- the commands to run at the scope (`pyclm check <dir>`, then
  `pyclm run <dir> --gui`);
- where the data goes: one `<position label>.zarr` per position (or
  `.hdf5` with `[output] format = "hdf5"`), plus `frames.parquet`,
  `events.parquet`, `status.json` and `log.log` in the directory;
- how to read it: `pyclm.io.open("<label>.zarr")` in Python
  (`documentation/data_format.md`), napari or Fiji for the OME-Zarr
  stores, or `pyclm export <dir>` for ImageJ hyperstacks;
- how to watch and steer a running experiment: `pyclm gui <dir>` and the
  command files (`documentation/command_line.md`: pause, stop, change a
  pattern's arguments).

Say plainly what has not been verified, such as presets checked without
the `.cfg` or a pattern never previewed on a real frame.
