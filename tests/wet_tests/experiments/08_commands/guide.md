# 08: commands from outside

## What must be generated live

The `PositionList.pos` of item 02 (copy it here).

## Steps

Two terminals. In the first:

```
uv run wet run 08
```

After the second timepoint has been acquired (watch the terminal), in
the second:

```
uv run wet commands 08
```

which sends, with pauses between them: `pause` (watch: t stops
advancing), `resume`, `set_exposure` 80 ms on the stimulation channel,
`set_property` Sola Power 40 (edit `wet/cli.py` `send_commands` if your
light source differs), `set_position` (a relative z nudge recorded as an
event), `set_pattern` bar_speed 2.0, and `stop_run`. The first terminal
then finishes early.

## Pass means

- the automatic checks: commands applied, events recorded with `source
  = "command"`, 80 ms on stimulation frames after the command, the run
  stopped before its last timepoint with `done: true` and an export;
- by eye: after `resume` the interval between timepoints is the same
  as before the pause (`scheduled_at` in the frames table).
