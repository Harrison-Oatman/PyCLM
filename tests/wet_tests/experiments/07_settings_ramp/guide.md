# 07: a method changes its own settings

## What must be generated live

The `PositionList.pos` of item 02 (one position is enough: keep only
`bar-1`, rename it `ramp-1`, or copy the two-position list and rename
both). If your light source is not a `Sola` with a `Power` property,
set `device` and `prop` in `ramp.toml` to a property that visibly
changes the stimulation frame's brightness.

## Steps

```
uv run wet run 07
```

## Pass means

- exposure and property requests applied (`events.parquet`), the
  stimulation exposure rising in the frames table, the property column
  present;
- by eye: the stimulation frames get brighter over the run.
