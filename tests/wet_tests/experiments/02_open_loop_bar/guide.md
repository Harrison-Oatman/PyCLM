# 02: the reference open-loop run

## What must be generated live

- **A sample**: anything fluorescent in the 545 channel (fixed cells, a
  fluorescent slide, beads).
- **`PositionList.pos` with two positions** named `bar-1` and `bar-2`:
  MicroManager → Devices → Stage Position List, focus at each (PFS on
  and locked on a Nikon so the PFS offset is stored), Save as
  `PositionList.pos` into this directory. Then close MicroManager.

## Steps

```
uv run wet run 02
```

Ten timepoints at 20 s: about 4 minutes. Watch the DMD through the
eyepiece or the camera live view once: a bar sweeping the field.

## Pass means

- the automatic checks in `results/02_open_loop_bar.md` all pass:
  completed without errors, on time, PFS locked at every visit, 10
  stimulation and 5 imaging frames per position, exposure as
  configured, a lit pattern, no large focus corrections;
- by eye: the bar in `imaging/pattern` of the viewer (`pyclm gui
  experiments/02_open_loop_bar`) lies on the lit cells of the
  stimulation frames;
- if a previous release's run of this same directory exists, compare
  `patterns/dmd/0` entry 0 of both: identical.
