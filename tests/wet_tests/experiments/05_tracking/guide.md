# 05: tracking

## What must be generated live

The cells and the `PositionList.pos` of item 04 (copy the list here).

## Steps

```
uv run wet run 05
```

## Pass means

- the checks of 04 pass, and at least three tracks are seen at two or
  more timepoints (`tracks.parquet`);
- by eye: in `imaging/tracks` of the viewer a cell keeps its colour
  across timepoints.
