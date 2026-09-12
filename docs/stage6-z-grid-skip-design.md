# Stage 6: grid acquisition and skipping idle positions — design and decisions

Date: 2026-09-11. Status: **implemented** 2026-09-11, every decision in §5
as recommended except one amendment from Harrison: patterns are kept in
**both** DMD and camera coordinates in OME-Zarr (§2.5, decision 9).
Implementation notes, in the order of §4: (0) `PatternReview` and the
SLM-coordinates path removed; `camera_roi` in `PyclmConfig`, set by the
Controller and preview, composed into the affine by `compose_affine`
(`core/manager.py`), the DMD footprint in `pyclm check`; OME-Zarr format 3
with `patterns/camera`, `dmd_index`, the `imaging` policy, HDF5 `dmd`
datasets gzip-compressed. (1) Skipping: `PlannedEvent.skippable`, the
handshake before the move, `MicroscopeProcess._blank_skips`, the `skipped`
routing kind carrying `SkippedAcquisition` to the writers, `EventDoneMessage.skipped`,
`position_skipped` events, `skipped` in `status.json` and the viewer's
status line. (2) Grids: `core/grid.py` (`group_tiles`, `GridGeometry`,
`geometry_for`, `stitch`, `cut`), `positions_from_pos` folding tiles,
`MicroscopePosition.tiles/grid/geometry`, the Controller computing the
geometry per position, the microscope's `_acquire_grid` (per-tile move
through the same mover, settle, per-tile DMD image, stitch), the SLM buffer
cutting into a per-tile DMD list with ids, the writer's stitched shapes and
per-tile DMD entries, the pattern process's stitched `pattern_shape`,
`PatternContext.grid()`, preview tiling, the virtual microscope's
nearest-position window, `pyclm check --pixel-size-um`. Two things to know:
on the virtual microscope the camera ROI is reported `ROI_FACTOR` (4)
times the TIF and the stitched pattern is at that reported scale while
the stitched frames are at the TIF's (the same pre-existing artefact as
for plain dry runs; `stitch` lays frames out at their own scale); and a
grid pays the settle time once per tile per channel. Tests:
`tests/test_camera_roi.py`, `tests/test_skip.py`, `tests/test_grid.py`,
`tests/test_grid_run.py` (SLM buffer, microscope, writer, check, a dry run
of a 2 × 2 grid), plus the storage tests for both pattern spaces.

Revised four times the same day with Harrison. The first draft also
covered z-stacks; they were dropped at the last revision (§0), and the
configuration of grids moved from the experiment TOML to MicroManager's
own position list (§2.1). Along the way the camera ROI, camera-space
pattern storage, and the removal of the pattern-review method were added
(§2.4, §2.5, §3b).

Two requests from Harrison, in his words: grid acquisition where "the
user will want the pattern method to return a pattern which follows the
shape of the entire grid"; and, if feasible without major structural
change, "skip positions if no imaging is being taken at the position or
the DMD is all zeros at the position".

The short version: both fit the existing structure. A grid is **one plan
event that the microscope executes as several moves and snaps**,
publishing **one stitched frame per channel per timepoint**, so the
router, the pattern dock, segmentation, tracking, the viewer and the
readers keep their one-frame contract and the writer only sees a bigger
frame. Skipping is a decision the microscope makes at the SLM handshake,
where the pattern is in hand, and costs nothing structurally once
`update_pattern` is ordered before the stage move.

## 0. Z-stacks: considered and dropped

The first drafts designed z-stacks as one event per channel executed as
several z steps, with a per-channel projection (`z_stack`, `z_project` in
the TOML) going to the pipeline and the stack stored in a sibling zarr
group; the stack geometry would have come from MicroManager's saved MDA
settings (`AcqSettings.txt`). Harrison dropped it: it needs an extra file
exported from MicroManager and complicates storage, while grids integrate
naturally through the position list. If stacks return, that design (in
this file's git history, revision of 2026-09-11) still fits the one-event
model; nothing in the grid or skip work below forecloses it, and the
stimulation frame would stay a single plane either way.

---

## 1. What exists (the parts these features touch)

| Piece | Where | State today |
|---|---|---|
| Plan template | `core/plan.py` `_template`, `events_at` | One `acquire` per (t, position, channel); index `{t, p, c}`. useq sub-sequence per position holds channels only; **no `grid_plan`**. `events_at` already skips a position with nothing due. |
| Event | `core/events.py` `AcquisitionEvent` | Position, exposure, binning, config; produces one 2-D frame. |
| Microscope | `core/microscope.py` | `position` event → `PositionMover.move_to` (PFS variant locks focus and leaves it locked; negates y). `acquire` → config, exposure, binning, wait until scheduled, `waitForSystem`, settle, one `snapImage`. `update_pattern` → SLM handshake, `setSLMImage`. Acks only acquisitions (`EventDoneMessage`). |
| SLM buffer | `core/manager.py` `SLMBuffer` | One `(pattern_id, uint8 DMD image)` per experiment; `pattern_to_slm` warps a camera-space 0–1 pattern through the single global affine. Seeded all-zeros. No blank test anywhere. |
| Pattern method | `core/patterns/pattern.py` | `pattern_shape` = camera ROI at stimulation binning; µm meshgrid is corner-origin camera frame; `PatternContext.raw()` etc. return one 2-D array. Output is not shape-checked. |
| Writer | `core/storage/ome_zarr.py`, `hdf5_v1.py` | NGFF axes `t, c, y, x`; one array per cadence group; frame address `(local t, channel)`. Progress advances when every saved dataset of a timepoint is written (`_update_progress`); HDF5 v1 pre-allocates every expected dataset. Frames table has `x, y, z`. |
| Positions | `directories.py` | `PositionList.pos` and `multipoints.xml` → `MicroscopePosition(x, y, z, label, extras)`; `GridRow/GridCol` ignored; `-` in labels becomes `.`. |
| Virtual microscope | `virtual_microscope/` | Frame chosen by exact (x, y) match. |
| useq 0.9.2 | | `Position.grid_row/grid_col` and per-position sub-sequences exist; a grid of listed tiles is best recorded as positions with those fields rather than a `grid_plan` (which describes a rule, not a list). |

---

## 2. Grid acquisition

### 2.1 Configuration: MicroManager's tile creator and position list

MicroManager's Stage Position List has **Create Grid** (the tile
creator). Verified against the current source, it writes into the
position list everything PyCLM needs, and it solves the orientation
problem itself:

- Each tile is a position labelled `<prefix>-<n>-<col>_<row>` with the
  prefix the user types in the dialog (default `Pos`), `<n>` a counter
  per grid created, and three-digit zero-padded column and row (for
  example `tissue-1-000_000`, `tissue-1-001_000`, …).
- Each tile carries `GridRow` and `GridCol` (the `.pos` property map
  keys PyCLM's own writer already emits), and the stage coordinates of
  every axis in the list: XY, the Z drive and, on the Nikon, the PFS
  offset, interpolated per tile between the corner points the user set.
- Tiles are generated in row-wise snake order.
- Tile spacing is computed from the camera image size, the pixel size,
  the overlap the user typed (µm, pixels or percent) **and the pixel-size
  affine** (`getPixelSizeAffine`, with the transpose/mirror correction),
  so the stage step for "one image to the right" is correct for the
  camera's orientation. This is MicroManager's Pixel Calibrator, run once
  per objective, and it is the reason no orientation flags are needed in
  PyCLM: tiles placed by `(GridRow, GridCol)` abut by construction.

PyCLM's position reader groups tiles by the label prefix before the final
`-<col>_<row>`: every position matching `^(.+)-(\d{3})_(\d{3})$` whose
`GridRow`/`GridCol` agree with the label is a tile of the grid named by
the prefix, and the grid becomes **one PyCLM position** with that prefix
as its label (`tissue-1` → `tissue.1`, so `tissue.toml` is its experiment,
exactly the existing naming rule). Its coordinates are the centre of the
tiles; it keeps the tile list (`MicroscopePosition.tiles`, each a full
position with its own z and PFS offset), `rows`, `columns`, and the tile
pitch in pixels derived from the stage spacing and the pixel size
(`pitch_px = |Δxy| / px`, so `overlap_px = w − pitch_px`; the check
requires a consistent pitch and a complete `rows × columns` rectangle, and
refuses a grid with deleted tiles). Positions without the pattern are
plain positions as today. Nothing is added to the experiment TOML for
grids. `multipoints.xml` is not extended (not needed, per Harrison).

### 2.2 The plan and the microscope

One `acquire` per channel per timepoint, as today, carrying the tile
list and the grid geometry (rows, columns, pitch). The
microscope loops the tiles inside that event: `mover.move_tile(core,
tile)` (Basic: `setXYPosition` and the tile's z; PFS: the same y
convention, the tile's PFS offset, then wait for the lock status), settle,
snap; then **stitches** the tiles into one frame by `(GridRow, GridCol)`
and the pitch (placement only: the later tile overwrites the overlap
strip; no registration, no blending) and publishes that one stitched
`AcquisitionData`. After the last channel's last tile
the stage is left where it is; the next position event moves it anyway.
The stitched frame's shape is `((rows−1)·pitch_y + h, (columns−1)·pitch_x
+ w)` at the channel's binning. In the useq sequence the grid is recorded
as the tiles' absolute positions (`useq.Position.grid_row/grid_col`)
under the grid position rather than as a `grid_plan`, since the tiles
come from the list, not from a rule.

**Channel-outer, tile-inner.** Each channel's event visits every tile
before the next channel does. This costs `tiles × channels` stage moves
per timepoint instead of `tiles`, but it is what makes "one stitched frame
per channel" a single event with a single acknowledgement, and adjacent
tile moves are short. The budget estimate charges `tile_move_seconds`
(config, default 0.5) + settle + exposure per tile per channel and warns
at the usual threshold.

**Stimulation across the grid.** The pattern method returns a pattern in
the stitched frame at stimulation binning (its `pattern_shape` becomes
the stitched shape, its µm meshgrid the stitched extent, so existing
methods work unchanged and "follow the shape of the entire grid"). The
SLM buffer cuts the stitched pattern into per-tile camera-space images
(each the full tile extent, so with `overlap > 0` the strip is lit from
both tiles; recommend `overlap = 0` for stimulation) and warps each
through the affine; it holds a **list** of DMD images per grid
experiment. The stim event carries the tiles; at each tile the microscope
sets the tile's SLM image, moves, snaps. One handshake per timepoint as
today (the reply carries the list).

`PatternContext.grid()` returns rows, columns, tile shape, overlap and the
tile stage coordinates for methods that want per-tile logic; nothing else
in the context changes. The SLM buffer starts checking the returned
pattern's shape against the expected one (a warning and a resize today's
code silently lacks), for grids and plain experiments alike.

### 2.3 Storage and tools

- The stitched frame is the channel's frame; group arrays take the
  stitched shape from the plan (`image_shape` learns about grids).
- Patterns are stored in both spaces (§2.5); for a grid experiment the
  camera-space one is the stitched pattern the method returned and the
  DMD images are one per tile; the viewer's overlay draws the stitched
  pattern over the stitched frame directly.
- Frames table: `x, y` are the reference point; the tile stage
  coordinates and geometry are written once to the group attrs
  (`pyclm.grid`).
- HDF5 v1: not supported for grid experiments (§2.5; check error).
- Preview: `--image` either the stitched size (used as is) or a tile-sized
  image (tiled `rows × columns`).
- Dry run: a `PositionList.pos` with tiles works as on the microscope;
  the simulated source matches a snap to the **nearest** listed position
  instead of exact (x, y); if the grid's TIF is larger than the frame, the
  tile is cropped at its offset from the grid centre (so a grid dry run
  with one large TIF per grid produces a coherent stitched frame);
  otherwise every tile is the same frame.
- Check: lists each grid (label, rows × columns, pitch, overlap in
  pixels); budget with tiles; warns when a grid has overlap and a stimulation
  frame.

---

### 2.4 The camera ROI (`pyclm_config.toml`)

The DMD does not cover the whole camera field. For a grid to be lit
continuously with `overlap = 0`, each tile must be exactly the region the
DMD can reach, so the camera ROI must be set to the DMD's footprint, in
MicroManager when the grid is created (Create Grid uses the current image
size for its spacing) and by PyCLM for the run. Today PyCLM only *reads*
the ROI the camera happens to have (`controller.py:186`). New:

```toml
# pyclm_config.toml
camera_roi = [x, y, width, height]   # unbinned camera pixels; omit to leave the camera as it is
```

- Applied by the run at start (`core.setROI`) and by `pyclm preview
  --snap`; recorded in the store metadata and the frames table.
- **The affine is in full-frame, unbinned camera coordinates** (the frame
  MicroManager's Projector calibration works in), as the binning scaling
  already assumes. `pattern_to_slm` composes the ROI offset before the
  binning scale (`translation += A · (x, y)`), so one calibration serves
  every ROI and binning. The same composition applies in preview and in
  the viewer's overlay.
- `pyclm check` computes the **DMD footprint in camera pixels** from the
  inverse affine (the DMD rectangle's corners), prints it, and suggests
  it as `camera_roi` when none is set. For a grid it derives the tile
  pitch from the stage spacing and compares it with the ROI and the
  footprint: pitch ≠ ROI size means the grid was created with a different
  image size (error); ROI larger than the footprint means strips of each
  tile can never be lit (warning). `pattern_shape` and `image_shape`
  follow the ROI as they do now.

### 2.5 Pattern storage: camera coordinates, and how much it costs

Today the OME-Zarr store keeps every distinct pattern as a DMD-space
image (`patterns/dmd/0`, `(N, 1140, 912)` uint8, zstd) and HDF5 v1 keeps
one `stim_aq/dmd` per timepoint, **uncompressed**. Harrison reports that
in an every-10 imaging, binning-4 experiment 75 % of the bytes are DMD
patterns. Measured with the writer's codec (zstd 5, bit-shuffle) on
representative patterns:

| Pattern | DMD-space 1140 × 912 | Camera full 2048² | Camera at binning 4, 512² |
|---|---|---|---|
| bars (period 300 px) | 1040 kB raw → **2 kB** | 4194 kB → 1 kB | 262 kB → **0.3 kB** |
| 400 cell discs | 1040 kB raw → **42 kB** | 4194 kB → 39 kB | 262 kB → **7 kB** |
| one binned imaging frame, 512² uint16 | | | 524 kB → ~260–350 kB |

So in OME-Zarr a pattern is already 1–15 % of one binned frame; the 75 %
comes from HDF5 v1's uncompressed `dmd` datasets (1 MB per timepoint,
raw). Two changes, one per format:

- **OME-Zarr stores patterns in both spaces** (Harrison's amendment,
  given how small they compress). `patterns/camera/0`, `(N, H, W)` uint8
  (0–255) at the stimulation binning, in the ROI frame, stitched for
  grids: what the method returned, one per distinct pattern id.
  `patterns/dmd/0`, `(M, H_slm, W_slm)` uint8 as today: what the DMD was
  given. For a plain experiment `M = N` and the two arrays are parallel.
  For a grid experiment every tile's DMD image is its own entry with its
  own DMD pattern id: the `rows × columns` images of one stitched pattern
  are stored **contiguously in tile order**, and the group's `pyclm`
  attrs carry three parallel lists, `pattern_ids` (DMD ids), `camera_ids`
  (the stitched pattern each came from) and `tiles` (`[row, col]`). The
  frames table keeps `pattern_id` / `pattern_index` for the camera-space
  pattern and gains `dmd_index`, the index of the first DMD image of that
  pattern (`dmd_index + k` is tile `k`); `events.parquet` stays about
  settings and commands. The stimulation frame of a grid is saved
  stitched like every other frame. The viewer's overlay and the export use
  the camera-space array directly (no inverse warp); `pattern_to_camera`
  remains for stores written before this stage. `pattern_policy` keeps
  `on_change` (every distinct pattern; recommended, since it is cheap)
  and `none`, drops `all`, and gains **`imaging`**: store only the pattern
  in force at timepoints where an imaging frame of the experiment is
  saved (`pattern_index` and `dmd_index` are null on the other
  stimulation rows). The `imaging` policy is the request's "only on
  frames when imaging actually happens"; the numbers say it is rarely
  needed with zarr, so it is an option, not the default. The store's
  format version becomes 3; `pyclm.io` reads 2 and 3.
- **HDF5 v1 keeps its layout** (`stim_aq/dmd`, DMD space, per timepoint)
  for the analysis code that reads it, and its `dmd` datasets gain gzip
  compression (chunked datasets under SWMR accept filters; readers are
  unaffected) — the actual fix for the 75 %. Grid experiments require the
  OME-Zarr format (`pyclm check` error): a stitched camera-space pattern
  does not fit the `dmd` contract.

## 3. Skipping idle positions

### 3.1 What already happens

`events_at` emits nothing for a position when no channel is due at `t`
(`plan.py:482`), so "no imaging at the position" already means no stage
move. The remaining case is the request's second half: something is due,
but it is only the stimulation frame and the pattern is blank.

### 3.2 Where the pattern is known

The Manager builds events before the pattern for `t` exists for open-loop
methods (a method with no requirements, such as the red/far-red programme,
generates at `t` itself), and the SLM buffer lives in another thread. A
Manager-side decision could therefore act on a stale pattern by exactly one
timepoint, which for a 1/0 programme string is wrong every other step.
The microscope, at the `update_pattern` handshake, holds the very pattern
that would be applied. That is where the decision goes.

### 3.3 Mechanism

1. **Order**: per position, `request_pattern`, then `update_pattern`
   (handshake), then `position`, then the acquisitions. Setting the SLM
   image before the stage moves is harmless (the DMD light path is off
   between stimulation exposures) and puts the pattern in hand before the
   move.
2. **Flag**: the plan marks the `position` event and the stimulation
   `acquire`/`update_pattern` events with `skippable_if_blank = True` when
   the position's due acquisitions at `t` are the stimulation frame only
   **and** no pattern requirement of that experiment is due on the
   stimulation frame at `t` (`stim_raw`/`stim_seg` in the method's
   requirements at a pattern-due `t`).
3. **Decision**: on a handshake whose pattern is all zeros (no lit pixel in
   any tile) and whose event is flagged, the microscope records
   `skip = (t, experiment)` and drops the flagged `position` and `acquire`
   events of that `(t, experiment)` as they arrive: no move, no exposure.
   The DMD keeps the blank image (already set). For each dropped
   acquisition it sends `EventDoneMessage(skipped=True)`.
4. **Accounting**: the Manager writes an `events.parquet` row
   (`kind = "position_skipped"`, `reason = "blank pattern"`) and counts it
   in `status.json` (`skipped`). If the stimulation frame was to be saved,
   the writer must not wait for it: the microscope also publishes a
   `SkippedAcquisition` item (kind `skipped`) through the router, which
   the writer subscribes to for every experiment; `_update_progress` and
   HDF5's `_timepoint_complete` treat it as written (the zarr slot stays
   zeros; HDF5's pre-allocated dataset stays empty and gains a `skipped`
   attribute). A frames-table row with `kind = "skipped"` keeps the table
   one-row-per-scheduled-frame for analysis.
5. **Settings**: `apply_settings` at the boundary is unaffected; a
   `set_position` request for a skipped position is applied at the next
   position event that runs.

Nothing about the plan's structure, the router or the storage layout
changes; the cost is one flag on events, one branch in the microscope,
one new data kind, and the reordering in (1), which is also the right
order for the grid stim (SLM images before the tile moves).

---

## 3b. Removing the pattern-review method

`PatternReview` (`pattern_review`, `core/patterns/pattern.py`) replays the
DMD images of an earlier HDF5 file; nobody uses it (Harrison). It is the
only subclass of `PatternMethodReturnsSLM`, which is the only reason
`CameraPattern.slm_coords` and the SLM-coordinates branch of
`SLMBuffer.pattern_to_slm` and `PatternProcess.run_model` exist. All of
it goes: the class, the registry entry, the base class, the flag and the
branches, `tests/test_pattern_method.py`'s construction test, and the
mentions in the method zoo, `data_format.md`, `architecture-notes.md` and
known issue #5 (marked removed). Every pattern method then returns camera
coordinates, which is what §2.5 stores.

## 4. Implementation order, tests, documentation

0. **Housekeeping**: remove `PatternReview` and the SLM-coordinates
   path (§3b); `camera_roi` in the config, applied at start and composed
   into the affine (§2.4, tests on the simulated core with an ROI and a
   preview through the affine); camera-space pattern storage with the
   `imaging` policy, HDF5 `dmd` compression (§2.5; storage tests updated,
   `to_dmd` round trip against the old warp).
1. **Skipping** (smallest, independent): plan order and flags; microscope
   branch and acks; writer `skipped` kind; Manager accounting; tests on the
   simulated core with a method that returns zeros on odd timepoints
   (frames table and events rows, no stage moves recorded by the simulated
   core on skipped steps, progress advances).
2. **Grids**: position-list grouping of tiles (unit-tested on a `.pos`
   written by MicroManager's tile creator, one to be taken from the scope);
   tiles on the event; mover `move_xy`; microscope tile loop and stitching (a pure
   `stitch(tiles, geometry, flags)` function, unit-tested with synthetic
   tiles including flips and overlap); pattern shape for grids; SLM buffer
   cutting and per-tile DMD list, handshake reply; writer stitched shapes
   and camera-space patterns; viewer overlay; preview tiling; dry-run
   nearest-position and cropping; check.

Documentation: the `pyclm_config` reference (`camera_roi`,
`pattern_policy = "imaging"`), the method zoo (pattern review gone),
`first_time_setup.md` (Create Grid, the Pixel Calibrator), `data_format.md`
(camera-space patterns, `skipped` rows, events row),
`custom_pattern_methods.md` (`grid()`, stitched
`pattern_shape`), `command_line.md` (check findings), the testing guide
(the Pixel Calibrator, a grid at the scope),
`architecture-notes.md`, CHANGELOG, `claude.md`.

Out of scope for this stage: z-stacks (§0); registration or blending in
stitching; per-tile pattern methods (the stitched frame is the
contract); grids with missing tiles; `multipoints.xml` grids.

---

## 5. Decisions to confirm

1. **One event, many snaps.** A grid is executed inside one acquisition
   event per channel and published as one stitched frame; `index` stays
   `{t, p, c}`. *Recommended*; the alternative (one event per tile, a `g`
   key in every index) touches every consumer and buys nothing the
   pipeline needs.
2. **Grids come from MicroManager's position list**, made with Create
   Grid and saved as `PositionList.pos`; a grid is recognised by the tile
   creator's label pattern plus `GridRow`/`GridCol`, and its label prefix
   names the experiment as for any position. No `[grid]` table.
   *Recommended* (revised: this is what Harrison asked for, and the file
   has everything, including per-tile z and PFS offsets).
3. **Orientation is MicroManager's**: the tile creator lays tiles out
   with the pixel-size affine, so PyCLM stitches by `(GridRow, GridCol)`
   and needs no flags. The one-time step is MicroManager's Pixel
   Calibrator per objective; the testing guide has it. *Recommended*
   (revised; replaces the `[stage]` flags).
4. **Channel-outer tile order** (each channel's stitched frame is one
   event; `tiles × channels` moves per timepoint). *Recommended* for the
   reason in §2.2; tile-outer would need per-tile events.
5. **Grid stimulation** = one stitched pattern from the method, cut per
   tile by the SLM buffer, applied per tile; patterns stored camera-space
   for grid experiments. *Recommended.*
6. **Skip at the handshake** in the microscope, with `update_pattern`
   ordered before the stage move, skipped acquisitions acknowledged and
   recorded (events row, frames row `kind = "skipped"`, status count),
   the writer told through a `skipped` data kind. *Recommended*; it is
   exact for open-loop programme methods, which a Manager-side decision
   is not.
7. **Skip condition**: stimulation-only due, blank pattern in every tile,
   and no method requirement on the stimulation frame at that `t`.
   *Recommended.* (An imaging channel due at the position always runs.)
8. **Camera ROI in `pyclm_config.toml`** (`camera_roi`, unbinned pixels),
   set by the run; the affine stays in full-frame coordinates and PyCLM
   composes the ROI offset; `pyclm check` prints the DMD footprint and
   checks a grid's pitch against the ROI. *Recommended.*
9. **Patterns stored in both spaces** in OME-Zarr: `patterns/camera`
   (stimulation binning, ROI frame, stitched for grids) and
   `patterns/dmd` (one image per tile for grids, contiguous per stitched
   pattern, ids and tiles in the attrs, `dmd_index` in the frames table);
   `pattern_policy` default stays `on_change` (measured: 1–15 % of a
   binned frame per pattern) with a new `imaging` option; HDF5 v1 keeps
   its layout and gains compression on `dmd`; grids require OME-Zarr.
   *Accepted as amended by Harrison* (the first draft dropped the DMD
   array).
10. **Remove `PatternReview`** and with it `PatternMethodReturnsSLM` and
   the SLM-coordinates return path. *Recommended.*
