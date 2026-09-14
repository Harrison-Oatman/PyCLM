# Wet tests before a release

*The checks that need a microscope. The suite (`uv run --group test
pytest`, ~260 tests) covers everything that can run on the virtual
microscope; this list is what it cannot: real hardware, real timing, real
files from MicroManager. Run it in order on the release candidate; each
item names what to run, what must be true, and what to keep as the
record. Estimated time: one afternoon with a prepared slide, plus one
overnight run.*

**The runnable form of this list is `tests/wet_tests/`** (its README says
how): one directory per item, a guide for what must be generated live,
and automatic verification. This page states what each item guards.

Record for each item: pass / fail, the date, the commit, and the
experiment directory (keep `log.log`, `status.json`, `events.parquet`,
`frames.parquet`). Put the summary in the release notes.

## A. The environment

1. **The core loads the lab's configuration.** `uv sync`, then
   `uv run mmcore info` shows the Micro-Manager directory and pymmcore
   11.2.1.71.0 / pymmcore-plus 0.14.0 (or whatever the lock says);
   `uv run pyclm check <dir>` prints the device-interface line and `ok`
   for `pyclm_config`. Loading must not report a device-interface
   mismatch. *Guards: the pinned pymmcore pair (docs/stage5b, "Device
   interface"), the real-core class (known issue #32).*
2. **The real core answers.** In the same check output, the MicroManager
   `.cfg` is read and presets are validated (no "presets not checked"
   line). Then `uv run pyclm preview <dir> <experiment> --snap`: one frame
   per channel with the presets applied, the raw frame the size of the
   camera ROI. *Guards: `RealMicroscopeCore` delegations, `camera_roi`.*

## B. One known experiment, end to end

3. **A short open-loop run matches the previous release.** The lab's
   reference directory (a bar experiment, two positions, imaging every
   2 timepoints, stimulation every timepoint, `steps = 10`), the same
   `pyclm_config.toml` as the last release plus `position_mover = "pfs"`.
   `uv run pyclm run <dir> --gui`. Must be true: no acquisition errors in
   `status.json`; no timepoint later than one interval; the frames'
   exposure and `Laser-Intensity` match the TOML (frames table); the DMD
   pattern matches the previous release's for the same timepoint (compare
   `patterns/dmd/0`); the viewer shows both positions and the pattern
   overlay lands on the lit cells. *Guards: the plan, the SLM handshake,
   per-channel overrides (known issue #31), the affine and binning.*
4. **The PFS locks at every position.** In `log.log` of item 3, one
   "move+focus took" line per position per timepoint and no PFS timeout;
   `events.parquet` has no `z_correction` larger than a few µm. *Guards:
   `PFSPositionMover`, the `position_mover` config key.*
5. **HDF5 format 1 still writes.** Item 3 again with `[output] format =
   "hdf5"`, `steps = 4`: the file opens in the old analysis notebook, the
   `dmd` datasets are gzip-compressed (`h5py` `.compression`), and
   `current_t_index` reaches 3. *Guards: format 1 compatibility.*

## C. Closed loop

6. **Cellpose segmentation and a per-cell method on live cells.** A
   closed-loop directory (`move_out` or the lab's own method) with
   `[segmentation] method = "cellpose"` on a dish of cells, `steps = 6`,
   imaging every 2. Must be true: `labels/segmentation/0` has cells at
   every imaging timepoint; the pattern lights the outward half of each
   cell in the overlay; the timing budget warning in `pyclm check` (if
   any) matches what `events.parquet` reports as `late`. *Guards: the
   router, the segmentation process, the GPU.*
7. **Tracking.** Item 6 with a `[tracking]` table: `tracks.parquet` has
   ids that persist across imaging timepoints for cells that did not
   move much (spot-check three). *Guards: the tracking process, the
   tracks label image.*
8. **Two segmentations of one channel.** The KTR example from the
   documentation (`[segmentation.nuclei]` and `[segmentation.cyto]`),
   `steps = 4`: both label images exist and the method's ratio is finite
   for most cells. *Guards: named segmentations, the measurement toolbox.*

## D. Runtime changes

9. **A method changes its own settings.** The documentation's
   exposure-ramp example on one position, `steps = 6`: the stimulation
   frames after the first change carry the new exposure and the laser
   property in the frames table, `events.parquet` has the requests with
   `t_applied ≥ t_requested`, and the hardware really changed (compare
   two stimulation frames' brightness). *Guards: `apply_settings`, the
   boundary rule, the overrides columns.*
10. **Commands from outside.** During item 3 (or a repeat), from a second
    terminal: `pause` (t stops advancing, the status line says paused),
    `resume` (the interval is preserved: compare `scheduled_at` before
    and after), `set_exposure`, `set_property` on a laser, `set_position`
    with a 2 µm z change, `set_pattern` with a bar-speed change, then
    `stop_run`. Each appears in `events.parquet` with `source =
    "command"` and takes effect from the next timepoint; the run ends
    cleanly with `done: true` and an ImageJ export. *Guards: the command
    protocol, `Manager.apply_command`.*
11. **The red / far-red switch.** The documentation's `RedFarRedSwitch`
    programme `"1100"` on one position, `steps = 8`: the stimulation
    frames alternate between the two presets as the frames table's
    channel-group column shows, and the DMD is all-on. *Guards:
    `set_config` at the boundary, stimulation-only experiments.*

## E. Stage 6

12. **Skipping.** A method returning zeros at odd timepoints (the
    `OddOff` example in docs/testing-guide-stage6.md) with imaging every
    2: at odd timepoints the stage does not move to that position and no
    exposure is taken (watch, and check `log.log` for "stage move
    skipped"); `frames.parquet` has `kind = "skipped"` rows and
    `status.json` counts them. *Guards: the handshake-before-move order,
    the blank test.*
13. **The camera ROI.** `pyclm check` prints the DMD footprint; with the
    suggested `camera_roi` in the config, a run's frames have the ROI's
    size and a `full_on` pattern lights the whole frame edge to edge in
    the stimulation frame. *Guards: `setROI`, `compose_affine`.*
14. **A grid.** A 2 × 2 grid from Create Grid (prefix = the TOML stem,
    overlap 0, the ROI of item 13 active) on a slide with recognisable
    features, `steps = 4`: the stitched frame is continuous across tile
    boundaries in both directions; `patterns/dmd/0` has 4 images per
    pattern with the right `tiles`; the PFS locks per tile; a bar pattern
    is continuous across tiles when viewed through the eyepiece or the
    camera live view. *Guards: `group_tiles`, `stitch`, `cut`, the
    per-tile DMD upload, the pixel calibrator.*

## F. Robustness

15. **Overnight.** The lab's standard experiment for at least 8 hours,
    two or more positions: `status.json` ends with `done: true`; lateness
    stays bounded (no drift in `events.parquet` `late` rows); the export
    completes; disk use is as expected (patterns a small fraction of the
    frames). *Guards: memory, the SWMR/zarr writers, the wall clock.*
16. **A hardware error does not kill the run.** During a short run,
    provoke one failure (unplug the DMD's USB for one timepoint, or set a
    laser property to an invalid value through a command): the microscope
    logs the error, `status.json` counts it, the run continues, and the
    next timepoint is normal. *Guards: the microscope's error guard, the
    acknowledgement path.*
17. **Restart safety.** Start a run into a directory that already holds
    outputs: it refuses before touching the hardware (`FileExistsError`
    named in the check). Then `--force` is *not* a bypass for this; move
    the outputs and rerun. *Guards: early `FileExistsError`.*

## G. Reading the results

18. **The viewer during and after.** `pyclm gui <dir>` while item 15
    runs: the position axis, follow-the-run, the minimap with the field
    of view, the status line's skipped and settings counts; after the
    run, the same directory opens and every timepoint is present.
19. **Export and the readers.** `pyclm export <dir>` on items 3, 6 and
    14: the hyperstacks open in ImageJ with the raw, label and pattern
    channels; `pyclm.io.open` on each store returns `frames`, `events`,
    `pattern_at`, `camera_pattern_at`, and, for the grid, `grid`.

## When something fails

Record the item number, the commit, the directory, and the log line or
traceback in `docs/known-issues.md` as a new numbered entry, and do not
release until it is fixed or explicitly waived in the release notes.
