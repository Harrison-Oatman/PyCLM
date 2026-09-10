# The `pyclm` command

One command with six subcommands covers a run from the first file to the
exported stacks:

```
pyclm new <dir>                 create an experiment directory from a template
pyclm check <dir>               report what is wrong with it
pyclm preview <dir> <experiment> --image frame.tif
                                run the segmentation and the pattern method on one image
pyclm run <dir> [--dry] [--gui] check it, then run it
pyclm export <dir> [channels]   write ImageJ hyperstacks for finished outputs
pyclm gui <dir>                 open the live viewer
```

`pyclm <dir> [--dry] [--gui]`, the form from before these subcommands
existed, still means `pyclm run`. `convert_hdf5s` and `gui` remain as
aliases of `export` and `gui`. Custom methods need the Python entry points
(`run_pyclm`, `check_directory`, `preview`), which take the same
dictionaries; see [custom pattern methods](custom_pattern_methods.md).

## `pyclm new`

```bash
uv run pyclm new my_experiment --template closed-loop --name cells
```

writes `cells.toml`, `schedule.toml`, `pyclm_config.toml` and a `README.txt`
that says what to edit next. `--template open-loop` gives a moving-bar
experiment with no segmentation; `closed-loop` gives Cellpose segmentation
and a per-cell pattern. Existing files are never overwritten.

## `pyclm check`

```bash
uv run pyclm check my_experiment
```

reads everything in the directory and prints one line per finding, each
naming the file and the table or key it concerns:

```
pyclm check my_experiment
  INFO    pyclm_config.toml: ok (my_experiment\pyclm_config.toml)
  INFO    schedule.toml: 60 timepoints every 60 s (60.0 min)
  ERROR   cells.toml [pattern]: unknown argument 'bar_sped' for method 'bar' (did you mean 'bar_speed'?); accepted: bar_speed, duty_cycle, period
  WARNING cells.toml [segmentation]: configured but the pattern method 'bar' does not ask for it; it will not run
  INFO    PositionList.pos: 3 position(s): cells.00, cells.01, cells.02
  INFO    cells.00: 12 frames over 60 timepoints
  1 error, 1 warning
```

What it checks:

- the three kinds of file parse and every key is known, in range and of
  the right type (all problems in a file are listed together);
- every position has an experiment file and every experiment file is used;
- every method name is registered and its arguments match the method's
  constructor, with a suggestion for a misspelled one;
- what the pattern method asks for exists: its channels, the
  `[segmentation]` and `[segmentation.<name>]` tables, `[tracking]`;
  a table nothing asks for is a warning;
- the config groups, presets and devices the files name exist in the
  MicroManager configuration, when the `.cfg` at `config_path` (or
  `--mm-config FILE`) can be read; otherwise a line says they were not
  checked;
- the timing budget: timepoints estimated to take longer than the interval;
- outputs that already exist.

`--dry` additionally copies the directory, shortens it to two timepoints,
and runs it on the virtual microscope, which needs TIF files in the
directory (see the dry run notes in [first-time setup](first_time_setup.md)).

An error means the run cannot start; `pyclm run` performs the same check
first and stops on errors (`--force` overrides, `--no-check` skips). A
warning means something looks unintended but the run can proceed.

## `pyclm preview`

```bash
uv run pyclm preview my_experiment cells --image snap.tif --pixel-size-um 0.33
uv run pyclm preview my_experiment cells.00 --snap
```

runs the experiment's segmentation(s) and pattern method once, on a TIF
(used for every channel the method needs) or on frames snapped from the
microscope with the channel's presets and exposure applied, and writes
into `my_experiment/preview/<label>/`:

| File | What |
|---|---|
| `raw_<channel>.tif` | the image the method saw |
| `labels_<channel>_<segmentation>.tif` | each segmentation it asked for; `..._tracks.tif` the tracked ids |
| `pattern_camera.tif`, `pattern_overlay.png` | the pattern in camera coordinates, and drawn in cyan over the image |
| `pattern_dmd.tif` | the pattern on the DMD, through the affine of `pyclm_config.toml` |
| `preview.json` | the method and its arguments, what it asked for, how long each step took, the fraction of the field lit, and any settings it asked to change |

The experiment is a position label from the position list or a TOML stem;
`--t N` sets the timepoint the method believes it is at (for methods that
depend on time). Without `--pixel-size-um`, a TIF is taken to be what the
directory's `dry_run.yml` says (`pixel_size_um`, `binning`; see the dry run
notes in [first-time setup](first_time_setup.md)), so the DMD pattern of a
preview matches the dry run's. The machinery is the run's own, so what preview shows is
what the run does.

## `pyclm run`

```bash
uv run pyclm run my_experiment
uv run pyclm run my_experiment --dry --gui
```

checks the directory, prints the report, and starts the run unless there
are errors. `--dry` uses the virtual microscope, `--gui` opens the live
viewer, `--config` names a `pyclm_config.toml` elsewhere.

## `pyclm export` and `pyclm gui`

```bash
uv run pyclm export my_experiment          # every experiment, every group
uv run pyclm export my_experiment 545 stim # only these channels or groups
uv run pyclm gui my_experiment
```

See [Data format](data_format.md) for what the export contains.

In the viewer, positions are an axis, not layers: each channel is one layer
stacked over (position, t), so the layer list has one entry per channel
however many positions there are, and contrast limits are one setting per
channel that applies to every position. Switch position with the slider,
the positions list in the dock on the right, a click on the minimap, or the
`[` and `]` keys; **Follow the run** keeps the viewer on the position the
microscope is acquiring. Contrast is normalised as frames arrive until you
move a slider, then it stays where you put it (**Auto-contrast** turns that
back on; **Normalise now** resets every channel once). The dock also shows
the status line and the minimap, refreshed from `status.json` every second.
Experiments with the same group name at different cadences get separate
layers (`imaging@5/545`).

## Commands to a running experiment

A run reads commands from files, so a script or a person with a text editor
can steer it. Write one JSON file into `<experiment dir>/commands/`; the run
picks it up at the next timepoint boundary, applies it, records it in
`events.parquet` (`kind = "command"`, with the settings it caused as further
rows, `source = "command"`) and moves the file to `commands/done/`.

| Command | Fields | Effect |
|---|---|---|
| `pause` | | hold before the next timepoint |
| `resume` | | continue; the clock is shifted by the paused time |
| `stop_run` | | end after the current timepoint |
| `stop_experiment` | `experiment` | end that experiment; the others continue |
| `set_exposure` | `experiment`, `channel`, `ms` | as a pattern method's `set_exposure` |
| `set_config` | `experiment`, `channel`, `group`, `preset` | as `set_config` |
| `set_property` | `experiment`, `channel`, `device`, `property`, `value` | as `set_property` |
| `set_position` | `experiment`, any of `x`, `y`, `z`, `pfs_offset` | as `set_position` |
| `set_pattern` | `experiment`, `parameters` | calls the method's `update(**parameters)`; the default sets existing attributes and refuses unknown names |

`channel` is a preset name or `"stimulation"`. For example:

```python
from pyclm.commands import write_command

write_command("path/to/experiment_dir", {"command": "set_exposure",
                                          "experiment": "bar10.00", "channel": "545", "ms": 80})
```

or, by hand, a file `commands/anything.json` containing
`{"command": "pause"}`. `status.json` reports `paused`, `stopping`,
`pending_commands` and `commands_applied`, and the viewer's status line
shows them.
