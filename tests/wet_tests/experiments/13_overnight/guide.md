# 13: overnight

## What must be generated live

- The lab's standard experiment: replace `standard.toml` with the real
  one if it differs (keep the file name or rename the positions to
  match), and `PositionList.pos` with the usual number of positions
  named `standard-1`, `standard-2`, …
- Live cells that survive the night.

## Steps

```
uv run wet run 13
```

500 timepoints at 60 s: 8 h 20 min. Leave the viewer open on a
second screen if you like (`pyclm gui experiments/13_overnight`).

## Pass means

- completed, no timepoint late, at least 8 h elapsed, outputs written
  (note the size: patterns should be a small fraction of the frames);
- in the morning: the viewer opens every timepoint, the export completes.
