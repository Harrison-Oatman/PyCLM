---
name: pyclm-run
description: Start a PyCLM run on the microscope, follow it, tell the person about flags, and chain the next step - with `pyclm run --detached`, `pyclm watch` and `--then`. Use when an experiment directory is ready and the person wants it run, asks how a run is going, wants runs back to back or overnight, or wants a run paused or stopped.
---

# Run an experiment and follow it

This skill starts from an experiment directory that `pyclm check` passes
(`pyclm-design-experiment` makes one). It does not decide whether the run
may go ahead: the lab's rule for that is in `lab.md` ("Approval before
running on cells"). Follow that rule. With no lab context, the person
asking for the run is the approval.

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

Then rehearse on the virtual microscope if nobody has yet:
`pyclm run <dir> --dry`. It catches method errors and timing overruns in
a minute.

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
(`setup_time_seconds`); a `stopped` flag means the process exited: read
the end of `detached.log` and `log.log` and tell the person.

Offer the live viewer: `pyclm gui <dir>` opens it on the running
directory, read-only.

## 3. Follow it

Start one watcher per run, in the background, and leave it running:

```bash
pyclm watch <dir> --images
```

It writes `watch/flags.json` every minute and appends to
`watch/flags_log.jsonl`. When the person asks how the run is going, read
`status.json` (timepoint, lateness, errors) and `watch/flags.json`, and
answer in two or three lines: timepoint of total, on time or how late, any
flags since the last answer.

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

- `pyclm watch` prints that the run is done; `status.json` says so too.
- Tell the person: timepoints acquired of planned, any flags, where the
  data are.
- Offer a preview movie: `pyclm export <dir> --movies` (one MP4 per
  experiment, the light pattern as a cyan overlay, never cropped).
- At the end of the session, use `pyclm-close-session`.
