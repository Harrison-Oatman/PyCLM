---
name: pyclm-run
description: Start a PyCLM run on the microscope, follow it, tell the person about flags, and chain the next step - with `pyclm run --detached`, `pyclm watch` and `--then`. Use when an experiment directory is ready and the person wants it run, asks how a run is going, wants runs back to back or overnight, or wants a run paused or stopped.
---

# Run an experiment and follow it

This skill starts from an experiment directory that `pyclm check` passes
(`pyclm-design-experiment` makes one, here or ahead of time). It does not decide whether the run
may go ahead: the lab's rule for that is in `lab.md` ("Approval before
running on cells"). Follow that rule. With no lab context, the person
asking for the run is the approval.

## Which PyCLM is this?

First, `pyclm --version` (PyCLM before 1.2 has no `--version`; then
`python -c "import importlib.metadata as m; print(m.version('closed-loop-microscopy'))"`
with that environment's Python) and `pyclm run --help`. This skill needs PyCLM
1.2 or later (`run --detached`, `watch`, the run lock and the run record).
With an older PyCLM:

- Tell the person which version you found and where (the full path of
  the `pyclm` you ran), and that it has no `--detached` or `watch`.
- Recommend installing 1.2 or later in its own folder
  (`pyclm-microscope-bringup`, step 1), not over an environment another
  project uses. Do not upgrade an install you did not make.
- Do not start a run as a hidden background process to make up for a
  missing `--detached`. If the person wants to run on the old version
  anyway, run it in a terminal they can see, with
  `PYTHONUNBUFFERED=1` and its output in a log file. Follow it by reading
  `status.json` and `log.log` in the directory, not its printed output.
  Tell them that closing that terminal stops the run.

## 0. A plan made ahead of time

People often design on their own computer and run on the scope later. A
plan is a folder: the experiment directory (or several), with its TOMLs,
and optionally a short plan note and a `pyclm_project.md` pointer. When a
session on the microscope PC starts in such a folder, or the person says
"run the plan in this folder":

1. Read the plan note and the pointer, and say in two lines what the plan
   is and which directories it runs, in what order.
2. Replace what only the scope can know: `pyclm_config.toml` from this
   microscope (its note says where), and today's positions (below,
   "Positions, focus and plate maps"). A plan's own copies of these are
   placeholders; never run with them.
3. `pyclm check` each directory against this scope (`--mm-config` if the
   `.cfg` is not the config's). Presets and pixel size are the usual
   mismatches between a plan and a scope.
4. Then step 1 below. Anything the check changed from the plan goes into
   the summary, so the go-ahead covers it.

## Positions, focus and plate maps

Ask before using a plate map or a stored focus. The person may want the
well that is in focus right now, and finding wells for a new map needs
their help.

**Plate maps.** `microscopes/<name>/plates/<format>.toml` in the lab
context holds where one vessel format's wells sit on this stage (two
corner well centers, the pitch) and its focus (z, lock offset), plus a
`[[confirmed]]` entry for each session that found a well where the map
put it (the template's `plates/example-96well.toml` has the fields).

- Ask which vessel is on the stage, if nobody has said.
- No map for it: ask "I can't find a <format> plate map for this scope.
  Would you like to help set up a calibration?" Setting one up needs the
  person to put one well in focus (or two, far apart, for the rotation)
  and say which wells they are. If they would rather use the well in
  focus now, use the stage's current position and make no map.
- A session writes a map directly, without a proposal, when it knows
  enough: the format (a 96-well and a 6-well plate both have a B3) and
  which well each reference position is. If either is a guess, ask, or
  make no map. A vessel format the lab will not use again needs no map.
- A map exists: go to the well the experiment needs only once the person
  has agreed (the focus question below), snap with no DMD light, and if
  the well is where expected add a `[[confirmed]]` entry (date, well,
  offset found minus predicted, z).
- Rely on the map once it has two or more confirmations from different
  days with offsets under a tenth of the pitch. Until then, check one
  well against it before placing fields, and say so.
- If the map is off (a well not where predicted by more than a quarter
  of the pitch), do not correct it silently: say so, find the well with
  the person, and write a new map dated today.

**Focus.** When a plate map or the microscope note has a z and lock
setting from a previous session, ask, as one question with choices:

> I have a height map stored from a previous session. Would you like me
> to reuse the z drive and PFS settings?
> 1. Yes, use the previous settings.
> 2. No, the sample is in focus right now.
> 3. No, open Micro-Manager so I can get the cells in focus.
> 4. Something else (say what).

Name the lab's focus lock in place of "PFS". For 1, set z and the lock,
engage it, snap and look at the frame; if it is not in focus, say so and
offer 3. For 2, read x, y, z and the lock offset from the stage and use
them. For 3, open Micro-Manager yourself (`launcher\Start-MicroManager.bat`
in the lab context, or the program the note names) and ask the person to
focus; when they close it, read x, y, z and the lock offset from the
stage. With nothing stored, ask whether the sample is in focus now or
Micro-Manager should be opened. The GUI and PyCLM cannot hold the
hardware together, so never open it while a run is going, and close
nothing the person has open.

**Fields.** Scouting around a focused position with no light, then
picking fields on the Positions page (`pyclm-session-pages`), is the
default. Write the picked positions as the experiment's `PositionList.pos`
(`pyclm.directories.write_position_list` writes one).

## 1. Before light goes on cells

Say these in a few lines, then wait for the go-ahead the lab's rule asks
for:

- what runs: experiments, positions, pattern methods, light channel;
- how long: timepoints x interval, plus `setup_time_seconds` and any
  `start_delay_seconds`; when it should end, in the person's local time;
- where the data go and roughly how big (positions x channels x
  timepoints x frame size);
- what `pyclm check` warned about, if anything.

If the lab context has a microscope note, read its "Check at session
start" list out first; a known fault (a light source not firing, a stage
error) changes the plan.

Then rehearse on the virtual microscope if nobody has yet. A dry run
takes as long as the real one and writes into the directory, so rehearse
on a copy: copy the directory, set `steps` to 3 and `interval_seconds` to
a few seconds in the copy's schedule, and `pyclm run <copy> --dry`. It
catches method errors and files the run cannot read in a minute; timing
is checked by `pyclm check`'s timing budget, not by the short copy.

## 2. Start it detached

```bash
pyclm run <dir> --detached
```

Always `--detached` when you start the run: an agent's shell, its process
tree, or the terminal can be closed without stopping the run. The command
checks the directory, starts the run in its own process, prints its process
id and writes `detached.json` and `detached.log`.

- If it refuses because the microscope is in use (a lock held by another
  run, in PyCLM versions with the run lock), say who holds it; never use
  `--force` to take it unless the person who holds it agrees.
- On Windows, if it says the terminal's job object would not let it go,
  the run survives the window closing but not the job. For a run that must
  survive anything, start the same command from a Windows scheduled task,
  and tell the person.

Within two minutes, check that it is acquiring:

```bash
pyclm watch <dir> --once
```

`no_status` right after the start is normal during setup
(`setup_time_seconds`), and `status.json` shows no acquired timepoint
until the first one ends. A `stopped` flag means the process exited: read
the end of `detached.log` and `log.log` and tell the person.

Offer the live viewer: `pyclm gui <dir>` opens it on the running
directory, read-only.

Make the Running page (`pyclm-session-pages`) once it is acquiring, and
rebuild it at each check. When positions are picked for a run, make the
Positions page before asking for them to be approved. Pages only show the
run; nothing on a page controls it.

## 3. Follow it

Start one watcher per run, in the background, and leave it running:

```bash
pyclm watch <dir> --images
```

It writes `watch/flags.json` every minute and appends to
`watch/flags_log.jsonl`. When the person asks how the run is going, read
`status.json` (timepoint, lateness, errors) and `watch/flags.json`, and
answer in two or three lines: timepoint of total, on time or how late, any
flags since the last answer. Rebuild the Running page with the same check.

How the person hears about a flag while they are away is the lab's
choice (`lab.md`, "Notifications"); it is usually a command for
`--on-flag`, which gets the path of `flags.json` as its last argument.
PyCLM sends no messages itself. Do not invent a notification channel; ask
if `lab.md` says nothing.

What each flag means and what to look at:

| Flag | First thing to look at |
| --- | --- |
| `stale`, `stopped` | the end of `detached.log`, then `log.log` |
| `late` | `log.log` timing lines; too many positions for the interval? |
| `acquisition_error`, `process_error`, `log_error` | the new `ERROR` lines in `log.log` |
| `dropped_frames` | disk space and write speed of the data drive |
| `brightness`, `saturation`, `focus` | the newest frames in `pyclm gui`; the focus lock |

A flag is information for the person, not a reason to act. Report it with
the evidence (the log line, the frame, the timepoint) and keep the run
going unless the person decides otherwise. The image thresholds are
starting points: a pattern that dims or brightens the field on purpose
trips them, and saying so once is enough.

## 4. Pause, change, stop

Runs are steered through command files, which the run polls between
timepoints (`documentation/command_line.md`, "Commands to a running
experiment"):

```python
from pyclm.commands import write_command

write_command("<dir>", {"command": "pause"})  # resume, stop_run, set_*, set_pattern
```

`stop_run` ends after the current timepoint and keeps everything
acquired. Send `pause`, `stop_run` or a change only when the person asks
for it, and say what you sent; `status.json` shows `paused` or
`stopping` once it has been read. Never kill the run's process.

## 5. Back to back and overnight

`--then` runs one command once the run says it is done and its files have
settled for 30 s:

```bash
pyclm watch <dir_1> --images --then "pyclm run <dir_2> --detached"
```

- Check every directory in the chain with `pyclm check` (and a dry run)
  before the first one starts; nobody is there to fix the second.
- The lab's approval rule covers each run in the chain: show the whole
  chain at once, so one go-ahead can cover it if the rule allows.
- For an export or a copy after the last run: `--then "pyclm export
  <dir> --movies"` or `--then "pyclm pack <dir> <destination>"`.
- Write the chain down (which directories, in what order, started when)
  in the session's notes, so a later session can see what was planned.

## 6. When it ends

- `status.json` says the run is done; `pyclm watch` says so at its next
  check, up to `--every-s` (60 s) later.
- Tell the person: timepoints acquired of planned, any flags, where the
  data are.
- Offer a preview movie: `pyclm export <dir> --movies` (one MP4 per
  experiment, the light pattern as a cyan overlay, never cropped).
- At the end of the session, use `pyclm-close-session`.
