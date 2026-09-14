# 04: closed loop with cellpose

## What must be generated live

- **Live cells** with a 545-channel marker dense enough to segment (a
  confluent-ish field, 50 to 300 cells).
- **`PositionList.pos` with one position** named `cells-1`, focused
  with PFS locked.
- The cellpose dependency group installed in the PyCLM checkout
  (`uv sync --group cellpose` there) and a GPU, or expect the timing
  warning from the check.

## Steps

```
uv run wet run 04
```

Six timepoints at 60 s. `pyclm check` will estimate the segmentation
time; if it warns that timepoints exceed the interval, note it and
compare with the `late` events afterwards.

## Pass means

- the automatic checks pass: cells segmented at every imaging
  timepoint, a pattern lighting part of the field;
- by eye in the viewer: the pattern is the outward half of each cell.
