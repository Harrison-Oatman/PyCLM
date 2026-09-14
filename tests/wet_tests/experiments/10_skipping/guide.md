# 10: skipping idle positions

## What must be generated live

The `PositionList.pos` of item 02 with the positions renamed
`oddoff-1` and `oddoff-2`.

## Steps

```
uv run wet run 10
```

Stand at the microscope: at odd timepoints the stage must not move to
either position and no exposure must be taken (no shutter, no frame).

## Pass means

- the automatic checks: skipped rows at odd timepoints, the events, the
  count in `status.json`, "stage move skipped" in the log;
- by eye: the stage stayed still at t = 1, 3, 5.
