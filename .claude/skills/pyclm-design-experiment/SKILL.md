---
name: pyclm-design-experiment
description: Turn a researcher's optogenetic experiment idea into a PyCLM experiment directory that passes `pyclm check` and runs on the virtual microscope. Use when someone wants to set up, plan, or rehearse a PyCLM run, choose a pattern method, write experiment/schedule TOMLs, or asks "how do I run X with PyCLM".
---

# Design a PyCLM experiment

Goal: a directory the researcher can take to the microscope, with every
file checked and a dry run that shows the pattern they meant. Do not
write a custom pattern method until the built-in ones are ruled out; for
that, use the `pyclm-write-pattern-method` skill.

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
  channel is segmented, and is Cellpose (needs a GPU) acceptable?
- **How many conditions, and how many positions each?** Each condition is
  one experiment TOML. Each position names its TOML by its label.
- **Duration and cadence.** The interval between timepoints, the number of
  timepoints, how often to image (`every_t`) and how often to stimulate.
- **Which microscope.** An existing `pyclm_config.toml` for that scope?
  If there is none and the scope has never run PyCLM, stop and use
  `pyclm-microscope-bringup` first. A dry run needs no microscope.
  Take channel and preset names from that scope's `.cfg`; every lab
  names them differently.

## 2. Pick the pattern method

Read `documentation/method_zoo.md` (or the method zoo page on
pyclm.readthedocs.io). Map the description to a built-in method and its
arguments. The common ones:

| Want | `method` |
|---|---|
| Field of moving stripes (negative speed reverses) | `bar`, `sawtooth`, `bar_bounce`, `rotate_bar` |
| Whole field on, or pulsed on a schedule | `full_on`, `global_pulse`, `pulse_schedule`, `global_cycle` |
| Light the outward / inward / lower half of each cell | `move_out`, `move_in`, `move_down` |
| Different methods on two halves of the field | `split` with `[pattern.left]`/`[pattern.right]` or `top`/`bottom` |
| A static image | `centered_image` |

The authoritative list is `known_models` in
`src/pyclm/core/patterns/__init__.py`. Read the class's `__init__` for
the argument names, units and defaults. `pyclm check` rejects unknown
arguments, so never guess them.

## 3. Write the directory

```bash
pyclm new <dir> --template open-loop --name <condition>    # or closed-loop
```

Then edit:

- `<condition>.toml`: one per condition. Channels are presets of the
  MicroManager config group, `[stimulation]` is the DMD event,
  `[pattern]` holds `method` plus its arguments, `[segmentation]` is only
  for closed loop. `[channels]` may be omitted for a stimulation-only run.
  Keys written after a table header belong to that table, so top-level
  keys (`t_delay`, `t_stop`) go above the first table.
- `schedule.toml`: `[timing]` `steps`, `interval_seconds`,
  `setup_time_seconds`, `time_between_positions`. Exactly one schedule
  per directory (`schedule.<label>.toml` is allowed).
- Positions: a MicroManager position list (`PositionList.pos` or
  `multipoints.xml`) whose labels are `<condition>.<anything>`.
  `<condition>` must equal the TOML's stem.
- `pyclm_config.toml`: copy the scope's working one. Do not invent
  `affine_transform` or `slm_shape_*`, and never copy them from another
  rig or from an example; they come from that scope's calibration. A dry
  run works with the template's values.

`documentation/experiment_tomls.md` has every key.

## 4. Check, preview, rehearse

```bash
pyclm check <dir>                      # add --mm-config file.cfg off the scope
pyclm preview <dir> <condition> --image frame.tif --pixel-size-um <um per pixel>
pyclm run <dir> --dry --gui
```

- Fix every error `check` reports, and read its warnings aloud to the
  researcher. The timing-budget warning means the positions do not fit
  in the interval.
- `preview` writes the pattern for one image. Open it with the researcher
  and confirm it is what they meant before going to the scope.
- A dry run needs TIFs in the directory, named by position label or
  condition stem, or mapped in `dry_run.yml`. Set `pixel_size_um` and
  `binning` there if the TIFs are not full-resolution camera frames.
  Without them the DMD pattern lands off the field.
- The open-loop lag: a pattern generated at timepoint t reaches the DMD
  at t+1. Explain this if the researcher counts timepoints.

## 5. Hand over

Tell the researcher:

- what each file sets;
- the commands to run at the scope (`pyclm check <dir>`, then
  `pyclm run <dir> --gui`);
- where the data goes: `<condition>.<nn>.zarr`, `frames.parquet`, `events.parquet`;
- how to read it (`pyclm.io.open`, see `documentation/data_format.md`,
  or `pyclm export <dir>` for ImageJ).

Say plainly what has not been verified, such as presets checked without
the `.cfg` or a pattern never previewed on a real frame.
