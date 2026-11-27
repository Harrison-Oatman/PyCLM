# The `pyclm` command

One command with nine subcommands covers a run from the first file to the
exported stacks:

```
pyclm new <dir>                 create an experiment directory from a template
pyclm check <dir>               report what is wrong with it
pyclm preview <dir> <experiment> --image frame.tif
                                run the segmentation and the pattern method on one image
pyclm run <dir> [--dry] [--gui] check it, then run it
pyclm run <dir> --detached      check it, then run it in its own process
pyclm watch <dir>               follow a run and flag what needs a person
pyclm export <dir> [channels]   write ImageJ hyperstacks for finished outputs
pyclm export <dir> --movies     write an MP4 preview movie per position
pyclm pack <dir> <destination>  copy a finished run as a few zip files, checked
pyclm stitch <dir>              one image of every position's newest frame, placed by stage
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

`pyclm new demo --demo` writes a directory that runs anywhere with
`pyclm run demo --dry`: two experiments (`bar`, open loop; `cells`, closed
loop with a threshold segmentation defined in the directory's
`demo_methods.py` and listed under `methods` in its `pyclm_config.toml`),
synthetic cell images, and a `dry_run.yml`. About thirty seconds; nothing is
downloaded.

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
  ERROR   cells.toml [pattern]: unknown argument 'speed_um_per_mn' for method 'bar' (did you mean 'speed_um_per_min'?); accepted: spacing_um, speed_um_per_min, start_min, stop_min, width_um
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
- the timing budget: timepoints estimated to take longer than the interval
  (a grid's tiles each count a move, the settle time and the exposure);
- the camera ROI against the region the DMD can light, and each grid in
  the position list (rows × columns, spacing; with `camera_roi` set and
  `--pixel-size-um` given, whether the tiles abut, overlap or leave gaps);
- outputs that already exist.

Pattern methods and arguments renamed in 2.0 (for example `direction` →
`move_sequence`, `rad` → `radius_um`, `duty_cycle` and `period` →
`width_um` and `spacing_um`) still run; each old name is a warning that
says what replaces it. `--fix` first rewrites the experiment TOMLs to the
current names, keeping comments and layout and the original file as
`<name>.toml.bak`, then checks:

```bash
uv run pyclm check my_experiment --fix
```

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

For a segmentation that reads several channels (`channels = [...]` in its
`[segmentation]` table), give one TIF per channel as `CHANNEL=TIF` (a channel not named uses the
first TIF given):

```bash
uv run pyclm preview my_experiment cells --image 545=nuclei.tif --image 638=membrane.tif
```

The experiment is a position label from the position list or a TOML stem;
`--t N` sets the timepoint the method believes it is at (for methods that
depend on time).

`--t START:STOP[:STEP]` previews a stretch of the schedule, `STOP` included:

```bash
uv run pyclm preview my_experiment bar --image snap.tif --t 0:240:5
```

The method runs at each timepoint in turn on the same image, keeping its
state between them as it would in a run (its previous patterns, the
histories it asked for, the tracker's ids), so a whole schedule of light
can be checked in seconds instead of a dry run that takes as long as the
experiment. Segmentation runs once, on the one image. Timepoints before
`t_delay` or after `t_stop` are left dark. `pattern_camera.tif` and
`pattern_dmd.tif` become stacks with one frame per timepoint, and two files
are added: `pattern_montage.png`, up to twelve evenly spaced overlays
captioned with the timepoint and the minutes since the start, and
`pattern_overlay.gif`, every overlay in turn. `preview.json` lists, for
each timepoint, whether the experiment is active, whether the stimulation
cadence (`every_t`) delivers light then, and the fraction of the field lit.
A closed-loop method sees the same image at every timepoint, so the series
shows its timing and memory, not how it responds to cells that move. For a grid position the method sees the stitched frame: a
tile-sized `--image` is tiled rows × columns, a stitched-size one is used
as it is, and `pattern_dmd.tif` holds one DMD image per tile. Without `--pixel-size-um`, a TIF is taken to be what the
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

**One run at a time.** A run on the microscope (not a dry run) holds a lock
while it runs, so a second `pyclm run` on the same computer refuses to start
and says whose run holds the microscope, from which directory and since
when. This covers another terminal, another person's agent and a second
`--detached` launch. The lock is the file `microscope.lock` in
`%PROGRAMDATA%\pyclm` on Windows, or in the system temporary folder
elsewhere; set `PYCLM_LOCK_DIR` to put it somewhere else, for example a
folder every user of a shared computer can write. A lock left by a run that
crashed is replaced by the next run, with a warning in `log.log`. The lock
and the run record name the person as `PYCLM_PERSON` gives it, when it is
set (several people on one Windows login), and the login name otherwise.

**The run record.** When a run ends, however it ends, it writes
`run_record.json` into the directory: start, end and duration, host and
user, the command, the PyCLM version (and git commit), the SHA-256 of
`pyclm_config.toml` and of the Micro-Manager `.cfg`, per experiment the
pattern method and its arguments, the segmentation and tracking methods and
the last timepoint acquired, the outcome (`finished`, `stopped` or
`crashed`, with the error), error and dropped-frame counts, flags from
`pyclm watch`, and every output with its size. It is the file to read, or to
copy into a project's records, to know what a run was without opening its
data (see [data format](data_format.md)).

### Runs that outlive the terminal: `--detached`

```bash
uv run pyclm run my_experiment --detached
```

checks the directory in the terminal, as above, then starts the run in a
process of its own and returns at once. Closing the terminal, ending its
process tree, or restarting a coding agent that typed the command does not
stop the run. Use it for any run you will not sit next to, and always when an
agent starts the run for you.

The command prints the run's process id and writes two files into the
directory:

| File | What it holds |
| --- | --- |
| `detached.json` | process id, the exact command, start time, where the output goes |
| `detached.log` | the run's console output: warnings, errors, a traceback if it crashes |

`log.log` is written as in any run. `--detached` cannot be combined with
`--gui`; open the viewer on the running directory with `pyclm gui <dir>`. To
stop a detached run, send it a `stop_run` command (below); this ends it
cleanly and keeps everything acquired.

On Windows the run is started outside the terminal's console and process
group, and outside its job object where the job allows that. If a job does
not allow it (some remote-desktop and agent environments), the command says
so: the run then still survives the terminal window closing, but not the job
being closed. For runs that must survive anything, start
`pyclm run <dir> --detached` from a Windows scheduled task.

## `pyclm watch`

```bash
uv run pyclm watch my_experiment
uv run pyclm watch my_experiment --once
uv run pyclm watch my_experiment --images --on-flag "notify.bat"
uv run pyclm watch my_experiment --then "python next_step.py"
```

follows a run from the files it writes anyway and flags what needs a
person. It checks every 60 s (`--every-s`) until the run is done; `--once`
checks once and exits with code 1 if anything is flagged. It never writes
into `commands/` and cannot change or stop a run, so it is safe to start,
stop and restart at any time, from any terminal, on a run that is going or
finished.

| Flag | Raised when |
| --- | --- |
| `no_status` | the directory has no `status.json` yet |
| `stale` | `status.json` was not rewritten for `--stale-min` minutes (10) and the run is not done |
| `stopped` | the process started by `--detached` has exited, and the run is not done |
| `late` | an experiment's last acquisition was more than `--late-s` seconds late (default: one interval) |
| `acquisition_error` | an experiment's count of failed acquisitions rose |
| `process_error` | a process's error count in `status.json` rose |
| `dropped_frames` | the writer dropped frames, or some could not be delivered |
| `log_error` | new `ERROR` or `CRITICAL` lines, or a traceback, appeared in `log.log` |
| `brightness` | with `--images`: a channel's newest frame has under 0.5 or over 2 times the median of its first |
| `saturation` | with `--images`: over 1 % of a frame's pixels sit at the camera's maximum |
| `focus` | with `--images`: a channel's newest frame is under half as sharp as its first |

Counts (errors, dropped frames, log lines) are flagged when they rise, so one
old error is reported once. The image thresholds are starting points; a
pattern that dims or brightens the field on purpose will trip them.

Each check prints one line and writes `watch/flags.json` (the latest check)
and appends to `watch/flags_log.jsonl` inside the directory (`--out` puts
them elsewhere). `pyclm check` and dry rehearsals ignore that folder.

`--on-flag COMMAND` runs a command whenever a flag appears that it has not
run for in the last `--quiet-min` minutes (30), with the path of
`flags.json` added as its last argument. PyCLM sends no messages itself:
the command decides how a person is told (an email, a phone push, a chat
message, a sound). `--then COMMAND` waits until the run says it is done and
its files have stopped changing for 30 s, runs the command, and exits with
its exit code; use it to start the next run or an export overnight.

## `pyclm export` and `pyclm gui`

```bash
uv run pyclm export my_experiment          # every experiment, every group
uv run pyclm export my_experiment 545 stim # only these channels or groups
uv run pyclm gui my_experiment
```

See [Data format](data_format.md) for what the export contains.

### Preview movies: `--movies`

```bash
uv run pyclm export my_experiment --movies
uv run pyclm export my_experiment 545 --movies --fps 15 --scale 0.5
uv run pyclm export my_experiment --movies --split
```

writes `movies/<experiment>.mp4`, one movie per position, instead of the
ImageJ stacks. Each movie frame is one imaging timepoint, with the
channels side by side in gray. The light pattern in force at that time is
a 40 % cyan overlay (`--no-pattern` leaves it out), and segmentation
outlines are drawn where the run saved them (`--no-outlines`). Contrast is
fixed per channel over the whole movie, so a real change in brightness
stays visible. Frames are never cropped or rotated.

Naming channels exports only those. `--split` writes one movie per
channel (`<experiment>_<channel>.mp4`), `--scale` resizes, `--fps` sets
the speed (10), and `--out` writes elsewhere. Movies are H.264 when
`ffmpeg` is on the PATH or `imageio-ffmpeg` is installed. Otherwise
OpenCV's encoder is used, which VLC plays but some Windows players do not.
Movies can be made while a run is going, from the frames written so far.

In the viewer, positions are an axis, not layers: each channel is one layer
stacked over (position, t), so the layer list has one entry per channel
however many positions there are, and contrast limits are one setting per
channel that applies to every position. Switch position with the slider,
the positions list in the dock on the right, a click on the minimap, or the
`[` and `]` keys; **Follow the run** keeps the viewer on the position the
microscope is acquiring. Contrast is normalized as frames arrive until you
move a slider, then it stays where you put it (**Auto-contrast** turns that
back on; **Normalize now** resets every channel once). The dock also shows
the status line and the minimap, refreshed from `status.json` every second.
Experiments with the same group name at different cadences get separate
layers (`imaging@5/545`).

## `pyclm pack`

```bash
uv run pyclm pack my_experiment E:/                 # -> E:/my_experiment/
uv run pyclm pack --verify E:/my_experiment         # check a copy
uv run pyclm pack --unpack E:/my_experiment D:/data # -> D:/data/my_experiment/
```

copies a finished run somewhere else, such as a USB drive, a network
share or a cloud folder. Each top-level folder (each `.zarr` store) becomes
one uncompressed `.zip`, so the copy is a few dozen large files instead of
thousands of chunk files. Everything else is copied as it is, and
`movies/` always stays loose so the movies can be watched where they land.
It writes `manifest.json` (every original file, its size, and the archive
that holds it), then checks the copy against it.

It refuses a run whose `status.json` does not say done, or whose files
changed in the last 10 minutes (the ImageJ export may still be writing);
`--force` copies anyway. It skips `preview/` and the ImageJ exports, which
duplicate the stores (`--keep-exports` copies them too). `--checksums`
also hashes every file, which takes about twice as long. `--no-zip`
copies file by file.

`--verify` checks a copy, packed or not: missing files, wrong sizes, files
not in the manifest, and, with `--checksums`, corrupted contents.
`--unpack` checks a packed copy, unpacks it, and checks the result.
Nothing is ever deleted: keep the original until the copy is verified where
it landed.

## `pyclm stitch`

```bash
uv run pyclm stitch my_experiment
uv run pyclm stitch my_experiment --channel 545 --every-s 60
uv run pyclm stitch my_experiment --orientation flip_ud
```

draws one overview of a run: each position's newest frame, placed where
the stage was when it was taken (from the frames table), outlined and
labeled. It writes `stitched/overview.png`, in stage coordinates: columns
follow stage +x and rows follow stage +y. With `--every-s` it redraws until
the run is done, which is a quick way to see a whole plate or dish fill in
during a scout. It draws the first channel unless given `--channel`, and
the longest side is 3000 pixels (`--max-px`).

How the camera image sits on the stage depends on the microscope. If
neighboring fields look mirrored or turned relative to each other, pass
`--orientation` (`identity`, `flip_ud`, `flip_lr`, `rot90`, `rot180`,
`rot270`, `transpose`). Two overlapping positions show which one is right;
note it in your setup notes.

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
shows them. Commands are read between timepoints, so when the run is behind
they take effect after the timepoints already sent; `acquired_t` in
`status.json` says where the microscope is. A command still unread when the
run ends is recorded in `events.parquet` as `not applied`.
