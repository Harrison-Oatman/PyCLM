# 12: a grid

## What must be generated live

- **Pixel Calibrator** run once for the objective (MicroManager →
  Plugins → Pixel Calibrator), so the tile creator knows the camera's
  orientation.
- The **camera ROI of item 11 active in MicroManager** when the grid is
  created (the ROI tool in the main window): Create Grid spaces tiles by
  the current image size.
- A **slide with recognisable structure** at the tile scale (a grid
  slide, scratched coverslip, a dense fixed sample).
- **Create Grid**: Devices → Stage Position List → Create Grid, prefix
  `grid`, overlap 0, mark two opposite corners so the grid is 2 x 2,
  Create; Save as `PositionList.pos` here. MicroManager writes
  `grid-1-000_000` … with `GridRow` / `GridCol`.
- Then:

```
uv run pyclm check experiments/12_grid --pixel-size-um <unbinned um/px>
```

  must print `grid.1: grid of 2 x 2 tiles … the tiles overlap 0.0 x 0.0 um`.

## Steps

```
uv run wet run 12
```

Four timepoints at 60 s; each visits four tiles per channel.

## Pass means

- the automatic checks: a 2 x 2 grid, continuity across the seams
  (correlation of the rows either side), four DMD images per pattern,
  PFS locked per tile;
- by eye, decisive: open the stitched frame in the viewer and look at
  the two seams; features must continue across them. A mirrored or
  transposed layout means the pixel calibrator is wrong for this
  objective;
- through the eyepiece: the bar is continuous across tiles.
