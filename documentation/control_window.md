# The control window

```bash
uv run pyclm control path/to/experiment_dir          # on the microscope
uv run pyclm control path/to/experiment_dir --dry    # on the virtual microscope
```

The control window prepares an experiment directory and drives a run from
the outside. It runs in its own process and talks to a run only through
files in the experiment directory, so nothing that happens in the window,
including a crash, can affect a run in progress. The live viewer
(`pyclm gui`) is the same kind of bystander: it reads the data as it is
written and can be opened, closed or restarted at any time.

Three tabs:

## Run

- **Check** runs the same check as `pyclm check` and shows the report.
- **Rehearse (dry)** and **Start run** launch `pyclm run` as a separate
  process (with `--dry` for the rehearsal); its output appears in the
  panel. A run does not start while the check reports errors.
- **Pause**, **Resume** and **Stop run** send commands to the running
  experiment (see below). Pausing holds the run before the next timepoint
  and shifts its clock, so the interval is preserved when it resumes;
  stopping ends the run after the current timepoint and finalises the
  outputs.
- **Commands to one experiment**: pick the experiment and channel, then set
  an exposure, a device property (for example a laser intensity), an
  absolute stage position, or a parameter of the pattern method; or stop
  that one experiment while the others continue.
- The status line and the **minimap** (positions in stage coordinates,
  coloured by experiment, the current position filled in, experiments with
  acquisition errors in red) update every second from `status.json`.

The window holds the microscope only while no run is active. Starting a
run releases it; when the run ends the window takes it back.

## Positions

On the microscope this tab shows the stage controls, snap and live and an
image preview (pymmcore-widgets); with `--dry` it drives the virtual
microscope. Below them is the position table with one column the
MicroManager list does not have: **experiment**, a dropdown of the
experiment files in the directory. The label is kept in the
`<experiment>.<number>` form the run expects.

- **Load list** reads the directory's `PositionList.pos` (or
  `multipoints.xml`).
- **Add current position** appends a row at the stage's position, for the
  experiment of the selected row (or the first experiment).
- **Move to selected** drives the stage there; **Remove selected** drops a
  row.
- **Preview here** snaps a frame at the current position with the selected
  experiment's first channel and runs its segmentation and pattern method
  on it (`pyclm preview`), writing the results under `preview/`.
- **Save PositionList.pos** writes the table in MicroManager's own format,
  which MicroManager can import and `pyclm run` reads.

## Files

One form per configuration file, generated from the schema: every key has
the widget its type calls for, its limits, and its description as a
tooltip. A method table offers the registered method names and, once one
is chosen, lists the arguments its constructor accepts; the arguments
table below it holds the values. Named tables (`[channels.<preset>]`,
`[segmentation.<name>]`) are added and removed with buttons.

**Validate** runs the schema and shows every problem; **Save** validates
and writes. Saving changes only what changed: tables you did not touch keep
their comments and layout, edited values are replaced in place, removed
keys are deleted. **New experiment file** writes a closed-loop template
under the name you give.

## Commands from anywhere

The window's run controls are ordinary files, so a script or a person with
a text editor can do the same. Write one JSON file into
`<experiment dir>/commands/`; the run picks it up at the next timepoint
boundary, applies it, records it in `events.parquet` (`kind = "command"`,
with the settings it caused as further rows, `source = "command"`) and
moves the file to `commands/done/`.

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
