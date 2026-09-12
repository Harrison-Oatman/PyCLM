# Testing guide: Stage 6 (grids, skipping, the camera ROI) on the virtual and the real microscope

*For Harrison, 2026-09-11. The dry part runs anywhere; the microscope
part needs the scope and about an hour. Do it after the Stage 5b guide's
§2.1 (the core loads) has passed.*

## 1. Dry

### 1.1 The suite

```bash
uv run --group test pytest -q tests/test_camera_roi.py tests/test_skip.py tests/test_grid.py tests/test_grid_run.py
```

The last file ends with a dry run of a 2 × 2 grid on a synthetic TIF whose
four quadrants have the values 1–4; the stitched frame must come back with
the quadrants where they were. If you want to see that run by hand,
`tests/test_grid_run.py::test_dry_run_grid` shows the recipe: a
`PositionList.pos` from the tile creator, one TIF the size of the grid
named after the grid (`bar10.1.tif`), `camera_roi` the size of one tile,
`dry_run.yml` with the TIF's pixel size.

### 1.2 The camera ROI in the check

```bash
uv run pyclm check <any experiment directory>
```

Read the two `camera_roi` lines: the region the DMD covers in camera
pixels (from the affine) and the suggested `camera_roi`. Put the
suggestion into `pyclm_config.toml` and check again: the line now says it
will be set. Make it deliberately larger than the footprint: a warning
says part of every frame can never be stimulated.

### 1.3 Skipping

Take the scratch directory of the Stage 5b guide (imaging every 2
timepoints, stimulation every timepoint), give one experiment a method
that returns zeros at odd timepoints:

```python
class OddOff(PatternMethod):
    name = "odd_off"
    def generate(self, context):
        v = 0.0 if context.t % 2 else 1.0
        return np.full(self.pattern_shape, v, np.float32)
```

(register it with `run_pyclm(..., pattern_methods={"odd_off": OddOff})`),
and run dry. Afterwards `frames.parquet` has `kind = "skipped"` rows for
the stimulation channel at the odd timepoints of that experiment,
`events.parquet` has `position_skipped` rows, `status.json` counts them,
and the viewer's status line shows "N skipped". Note the one-timepoint
subtlety: an open-loop method's pattern for `t` is generated at `t`; the
handshake takes whatever the SLM buffer holds when it fires, so on a slow
machine the skip can land one timepoint later than the method's zeros.
That is inherent (the design notes explain why the decision is made at the
handshake) and harmless.

## 2. At the microscope

### 2.1 Pixel calibrator (once per objective)

MicroManager → Plugins → Pixel Calibrator, for the objective you will use
with grids. This writes the pixel-size affine the tile creator uses to
space tiles; without it, tiles may be mirrored or transposed relative to
the stage and the stitched frame will not be continuous.

### 2.2 The camera ROI

1. `uv run pyclm check <dir>` → copy the suggested `camera_roi` into
   `pyclm_config.toml`.
2. In MicroManager, set the same ROI on the camera (the ROI tool in the
   main window, or the camera's properties) so that Create Grid spaces the
   tiles by it.
3. `uv run pyclm preview <dir> <experiment> --snap`: the raw frame in
   `preview/` must have the ROI's size and the DMD overlay should reach
   every edge.

### 2.3 Create a grid

Devices → Stage Position List → Create Grid:

- prefix = the experiment's TOML stem (for example `tissue`),
- overlap 0 (µm, px or %), pixel size as calibrated,
- mark the corners of the region on a slide with recognisable features
  (a grid slide, or a scratched coverslip), 2 × 2 is enough,
- create, then Save as `PositionList.pos` in the experiment directory.

```bash
uv run pyclm check <dir> --pixel-size-um <unbinned pixel size>
```

should print `tissue.1: grid of 2 x 2 tiles, spacing … um; with camera_roi …
the tiles overlap 0.0 x 0.0 um`. A warning about gaps means the grid was
created with a different ROI or binning than the check assumes.

### 2.4 A short grid run

Set `steps` to about 5 and run:

```bash
uv run pyclm run <dir> --gui
```

- The viewer shows one frame per channel the size of the whole grid. Look
  at the tile boundaries: features must be continuous across them. A
  mirrored or transposed layout means the pixel calibrator step is wrong
  for this objective.
- With a full-on or bar method, the DMD lights every tile in turn: watch
  one timepoint through the eyepiece or the camera live view if you can.
  `patterns/dmd/0` in the zarr has 4 images per pattern and the `tiles`
  attribute lists their rows and columns.
- On the Nikon, each tile move goes through the PFS mover: the log shows a
  focus lock per tile. If a lock times out, the interpolated PFS offsets of
  the tiles are wrong; recreate the grid after focusing at each corner.
- The frames table's `x`, `y` for the grid are its centre; the minimap
  draws the grid as one position with the stitched field of view.

### 2.5 A skip on hardware

Run the Stage 5b scratch experiment with the `OddOff` method above at the
scope for a few timepoints: at odd timepoints the stage should not move to
that position and no exposure should be taken (no shutter click, no
frame); the run's log says "stage move skipped" and "stimulation skipped".

### 2.6 What to report back

For 2.2: the ROI and footprint lines. For 2.4: a screenshot of the
stitched frame at a tile boundary, the DMD image count, and whether the
PFS locked per tile. Anything that surprised you.
