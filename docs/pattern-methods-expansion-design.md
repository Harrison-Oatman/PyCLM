# Pattern-method expansion (2.0): design note

Status: approved by Harrison on 2026-10-26. PR 1 (the keyword
convention, renames, `start_min` / `stop_min`, `pyclm check --fix`) is
merged. PR 2 adds every approved method (§6); the per-family PRs planned
in §5 went in as one PR with one commit per family.

The proposal behind it, with the full list of candidates and why some
were declined, is in the project files as
`zoo/pattern-methods-proposal.md`. It had three sources:

- the lab's own methods in `toettchlab_pyclm/experiment_scripts`;
- the missing-methods list in [method-zoo-design.md](method-zoo-design.md) §4;
- new methods that use the runtime-change API (`PatternContext.set_position`,
  `set_exposure`, `set_config`, `set_property`).

New methods are added with the `pyclm-add-builtin-pattern` skill.

## 1. Approved methods

| Name | Family | Loop | Keywords | Source |
|---|---|---|---|---|
| `relative_clamp` | per_nucleus | closed + tracks + history | `channel`, `seg_channel = channel`, `calibrate_min = 30`, `targets = [0.5]`, `step_min = 60`, `min_range = 0.5` | lab (`DynamicRangeControl`) |
| `game_of_life` | *new* `diagnostic` | closed (stimulation frame) | `cells = 50`, `fill = 0.7`, `threshold`, `seed = 0` | lab; a DMD-calibration check on a fluorescent slide, listed apart from the experimental methods |
| `dose_grid` | static | open | `rows = 2`, `cols = 2`, `doses = [1.0]`, `gap_um = 50`, `spot_um = none`, `dose_mode = "gray"`, `period_min = 10` | lab (quadrants) + zoo `spot_array` |
| `pulsed` | timed | wraps; open | `period_min`, `on_min`, `phase_min = 0` | lab (`CircleDutyCyclePattern`) |
| `rectangle` | static | open | `width_um`, `height_um`, `x_um = 0`, `y_um = 0`, `angle_deg = 0` | zoo |
| `image_mask` | static | open | `path`, `threshold = none`, `fit = "center"` | zoo (`image`) |
| `chirp` | timed | open | `period_start_min`, `period_end_min`, `duty_cycle = 0.5`, `sweep_min`, `sweep = "log"` | zoo |
| `all_cells` | per_cell | closed | `channel`, `invert = false`, `dilate_um = 0` | zoo (+ `background`) |
| `random_cells` | per_cell | closed + tracks | `channel`, `fraction = 0.5`, `seed = 0` | zoo |
| `select_cells` | per_cell | closed | `channel`, `by` (`"area"`, `"intensity"`, `"distance_center"`, `"distance_edge"`), `min`, `max`, `top_n`, `band_um` (light a pixel band at the edge instead of whole cells) | zoo (+ `colony_edge`) |
| `nucleus_clamp` + `controller` | per_nucleus | closed (+ history) | `controller = "bang_bang"` / `"p"` / `"pi"`, `kp`, `ki`, `dose_mode` | zoo (`nucleus_proportional`); renamed in PR 1 |
| `population_clamp` | per_nucleus | closed + history | `channel`, `target`, `controller = "pi"`, `kp`, `ki`, `actuator = "duty"` / `"exposure"` / `"property"`, `device`, `property`, `min_value`, `max_value` | zoo; its actuator changes an imaging property |
| `recenter` | *new* `adaptive` | wraps; closed | `channel`, `deadband_um = 50`, `max_step_um = 200` | stage position |
| `drift_correct` | adaptive | wraps; closed (raw, history) | `channel`, `deadband_um = 2`, `max_step_um = 50`, `reference = "first"` | stage position |
| `auto_exposure` | adaptive | wraps; closed (raw) | `channel`, `percentile = 99`, `target = 0.6`, `min_ms = 5`, `max_ms = 1000`, `max_change = 0.25` | imaging property |
| `settings_schedule` | adaptive | wraps; open | `channel = "stimulation"`, `setting`, `group` / `device` / `property`, `steps = [[start_min, value], ...]` | imaging property |

Declined: `blinking_bar alternate`, `ramp`, `moving_spot`,
`random_pulses`, `colony_edge` as its own method (folded into
`select_cells`), `embryo_band`, `follow_cell`, `focus_track`.

## 2. Keyword convention (approved)

- **K1.** Every quantity with a unit names the unit as a suffix: `_um`,
  `_min`, `_ms`, `_deg`, `_um_per_min`, `_deg_per_min`. Time is always in
  minutes.
- **K2.** Fractions are 0–1 and have no suffix. `duty_cycle` means a
  fraction of *time*; a spatial lit fraction is `fill`. Stripes are given
  as `width_um` + `spacing_um`.
- **K3.** The sign of a speed reverses the motion. Cardinal directions are
  `u`/`d`/`l`/`r` (row 0 is up, and cells move towards the light).
  `angle_deg` is measured from the right, counterclockwise. No ±1
  integers.
- **K4.** `channel` is the measured channel and is required.
  `seg_channel` is used when the segmentation comes from another channel.
- **K5.** Setpoints are `target`; ranges are `min_*` / `max_*` or
  `target_min` / `target_max`, in camera counts.
- **K6.** Files are given as `path`.
- **K7.** Booleans are positive switches that default to false.
- **K8.** Anything random takes `seed`.
- **K9.** Every method accepts `start_min` / `stop_min`.
- **K10.** Schedules are lists of `[start_min, value]`.
- **K11.** Names are snake_case. `move_*` is per-cell migration,
  `*_clamp` is feedback to a setpoint, `embryo_*` is the embryo family;
  composites are short nouns or verbs. US spelling throughout.

## 3. Decisions

1. Adopt the convention for new methods and rename the old ones in the
   same release. **Approved.**
2. Old names stay as aliases with a `pyclm check` warning through 2.x and
   are removed in 3.0; `pyclm check --fix` rewrites TOMLs. **Approved.**
3. Setting and position behaviors are wrapping composites
   (`[pattern.inner]`, built on `split`'s `nested_methods` /
   `bind_registry`). **Approved.**
4. Add `camera_to_stage(dy_px, dx_px)`, from MicroManager's
   pixel-size affine (pixel size × identity in dry runs). Built on the
   method (`PatternMethod.camera_to_stage`), not the context: the method
   knows its binning. On a grid, an
   x/y move shifts every tile by the same offset; today it changes only
   the grid center, which the microscope ignores. **Approved.**
5. Graded doses: the DMD renders gray levels, so the default
   `dose_mode` is `"gray"`, with `"dither"` and `"duty"` as options.
6. `start_min` / `stop_min` live in the base class. **Approved; done in
   PR 1.**
7. New zoo families: `adaptive`, and `diagnostic` for `game_of_life`.
   **Approved.**
8. The merges (`dose_grid`, `nucleus_clamp` controllers, `all_cells
   invert`, `chirp`, `select_cells` edge band). **Approved.**
9. **Changed in review:** stage offsets are passed to the tracker. When
   the Manager applies an x/y change for an experiment, the tracking
   process is told the offset in pixels and shifts the previous centroids
   before linking, so track ids survive a move made by `recenter` or
   `drift_correct`.
10. The per-cell `move_*` family (`direction` → `move_sequence`,
    `rotate_ccw` → `move_rotate`). **Approved; done in PR 1.**
11. `focus_track` is declined.
12. The lab's downscaled-cellpose segmentation stays out of this work.
13. Release: the work ships as an **experimental 2.0 pre-release**, not a
    stable major release.

## 4. How PR 1 is built

- **Migration happens where a method is built.** `PatternMethod` has a
  metaclass whose `__call__` passes the keyword arguments through
  `cls.migrate_kwargs` before `__new__` / `__init__`. The dispatching
  bases (`bar`, `rings`) therefore see current names, and the run,
  `pyclm check`, `preview`, the zoo and `split` all agree without each
  translating. A plain rename is a `renamed_kwargs` entry, merged down the
  MRO. A value conversion (`period_h` × 60, `duty_cycle × period` →
  `width_um`, the `rotate_bar` speed → degrees with the sign flipped to
  counterclockwise) overrides `migrate_kwargs`. A default that became
  required is a `legacy_defaults` entry; it sits on the old-name aliases,
  plus the embryo methods, whose names did not change. A run logs one
  warning per class and note.
- **Renamed methods** are subclasses made by
  `renamed_method(cls, old_name, **legacy_defaults)`. Each has
  `deprecated` (so `pyclm check` warns), `renamed_to`, and a zoo
  `same_as` entry.
- **`pyclm check`** migrates each pattern table (and each `split` half),
  warns once per note, reports giving an old and a new name together as
  an error, and validates the migrated arguments. `_named_parameters` now
  follows `**kwargs` up the MRO, so arguments a method passes to its
  parents are accepted (`start_min`; also the per-cell `voronoi`, which
  it used to reject).
- **`pyclm check --fix`** (`src/pyclm/fix.py`) edits line by line: a plain
  rename keeps its line, value and comment; converted values are written
  where the first old key was. A value spanning lines, or an inline
  table, is reported and left alone. The original is saved as
  `.toml.bak`, and a file that would not parse is never written. Only
  built-in methods are migrated; a lab's method can declare
  `renamed_kwargs` for its own renames and is migrated at run time.
- **`start_min` / `stop_min`** are on `PatternMethod.__init__`.
  `render(context)` returns a dark field outside the window, and callers
  use it instead of `generate`. `global_pulse`, `blinking_bar` and
  `pulse_schedule` read `self.start_min` for their phase, as before.
- **The embryo fix:** see known issue 38.

### How it was checked

- 22 old-style configurations (every renamed keyword, both dispatching
  bases at speed 0 and non-zero, the aliases, embryo methods on a square
  field) were rendered at six times by `origin/main` and by this branch.
  All are pixel-identical (script in the PR description).
- `tests/test_keyword_migration.py`: equivalence of old and new keywords,
  `check` on the migrated arguments, aliases, required channels,
  `move_rotate reverse`, the `spokes` turning direction, the
  `start_min`/`stop_min` window, the non-square embryo field, `--fix`
  (renames, conversions, a `split` half, idempotence, values it refuses),
  and the archived `resources/tomls` still building.
- The test fixtures and wet-test TOMLs were rewritten by `--fix` itself.

## 5. PR plan

1. **Conventions and renames** (this note §4). Done.
2. **Static and timed:** `rectangle`, `dose_grid`, `image_mask`,
   `pulsed`, `chirp`, and a shared dose helper (`gray` / `dither` /
   `duty`).
3. **Per-cell and per-nucleus:** `all_cells`, `random_cells`,
   `select_cells`, `nucleus_clamp` controllers, `population_clamp`,
   `relative_clamp`.
4. **Adaptive:** `camera_to_stage`, the grid-tile shift, stage offsets to
   the tracker, `recenter`, `drift_correct`, `auto_exposure`,
   `settings_schedule`, and the `adaptive` zoo family. A wet test of x/y
   moves comes before the release.
5. **Diagnostic:** `game_of_life` and the `diagnostic` zoo family.

## 6. How PR 2 is built

Every method in §1 is in; the families are separate commits.

**Shared pieces.**

- `core/patterns/dose.py`: `deliver(dose, dose_mode, t_minutes,
  period_min)`. `"gray"` returns the dose, `"dither"` an 8 × 8 Bayer
  ordered dither, `"duty"` lights a pixel for the first `dose` fraction of
  every `period_min`. Used by `dose_grid`, `nucleus_clamp` (p / pi) and
  `population_clamp`.
- `WrappingPattern` in `composite.py`: one `[pattern.inner]` table,
  `nested_methods` / `bind_registry` as in `split`, the requests the union
  of its own and the inner method's (`merge_requests`, now shared with
  `split` and used by every method), `configure_system` / `update_binning`
  forwarded, `update` forwards `inner.<name>`. `inner_required` is true
  for `pulsed` and false for the adaptive wrappers (no inner method: a
  dark field).
- `CameraProperties.camera_to_stage`: the linear part of
  `core.getPixelSizeAffine()` read by the Controller
  (`camera_to_stage_matrix`; None when absent or singular), passed on to
  grid experiments by the pattern process and to the tracking process.
- Grid moves: `Manager._apply_change` shifts every tile by the same
  x / y / z delta and sets `pfs_offset` on every tile.
- Tracker offsets (decision 9): `AcquisitionEvent.stage_xy` snapshots the
  stage x / y when the event is made (the position object is shared and
  mutated by later moves). `TrackingProcess` keeps the last `stage_xy` per
  experiment and channel; on a change it converts the stage delta to frame
  pixels (inverse of `camera_to_stage` at the event's binning, or the
  pixel size) and calls `TrackingMethod.shift(-dy, -dx)` before `track`.
  `CentroidTracker.shift` moves its remembered centroids; the base does
  nothing.
- Zoo: families `adaptive` and `diagnostic`; `ZooMeta.pending` marks a
  card that waits for sample data (listed with its TOML under "Cards
  still to come"); a card shows the setting changes its example requests.

**Choices made while building (open to review).**

1. `nucleus_clamp` and `population_clamp` also take `period_min` (the
   period of a `"duty"` dose), like `dose_grid`.
2. `population_clamp`'s `actuator = "duty"` gives the whole field the
   controller's level as a dose, through `dose_mode` (gray by default).
3. The controllers work on the relative error `(target - value) /
   target`, so `kp = 1` means full light for a dark nucleus whatever the
   camera counts; the PI integral is held in 0–1 (no wind-up). `"pi"` in
   `nucleus_clamp` turns tracking on, since the integral belongs to a
   nucleus.
4. `relative_clamp` lights the whole field in its lit phase, keeps the
   last target after the list ends, and is bang-bang against the goal on
   whichever side light pushes the reading (a reporter that light dims
   works too). It corrects the lab version's phase arithmetic (§ known
   problems in the proposal).
5. `select_cells`: `min` / `max` are in the unit of `by` (µm², camera
   counts or µm); `top_n` keeps the largest values and a negative `top_n`
   the smallest (`top_n = -1, by = "distance_center"` is the cell nearest
   the center); `band_um` needs `by = "distance_edge"` and lights pixels,
   not cells. The colony is the cells' union, closed by 3 px and
   hole-filled; the image border is not an edge.
6. `random_cells` draws each track from `seed` and its id, so the draw is
   stateless and repeats in a rerun.
7. `pulsed` renders its inner method at every timepoint, so a closed loop
   inside keeps its state while the light is off.
8. `recenter` / `drift_correct` shift the pattern they return by the
   pixel equivalent of the move they request: the pattern is shown after
   the move. `drift_correct reference = "previous"` sums frame-to-frame
   displacements; the move shows up in the next displacement, so the sum
   corrects itself.
9. `auto_exposure` takes `full_scale` (the camera's largest count;
   default the image type's maximum), since a 12-bit camera in a 16-bit
   image never reaches 0.6 of 65535.
10. `settings_schedule` steps are minutes from `start_min` (like
    `pulse_schedule`), and a value is requested once, when its minute is
    reached: it takes effect from the next timepoint.
11. `game_of_life`: `cells` squares per side, `fill` the width of a lit
    square as a fraction of its pitch, `threshold` required (it depends
    on the slide). A square is read as alive when its mean is above
    `threshold` and above its border's mean (the lab's test).
    `agreement` and an info log line report how much of the board came
    back as projected.

**Zoo cards waiting for data** (`pending`), to be made from a run:

- `relative_clamp`, `population_clamp`: a time series of a nuclear
  reporter through dark and lit phases (whole-field light).
- `recenter`: a time series of a migrating colony.
- `drift_correct`: a time series with drift.
- `game_of_life`: frames of a fluorescent slide under the projected
  boards.

**Before the release:** a wet test of x / y moves on the microscope
(`recenter` on a slide), to confirm the sign of the pixel-size affine
against the stage.

### How it was checked

- `tests/test_new_pattern_methods.py`: each method's rule (which pixels
  light, when), its argument checks, the dose modes, the controllers'
  arithmetic, the clamps' phases, the wrappers' merged requests and
  `update`, `pulsed` around `full_on` equal to `global_pulse`,
  `camera_to_stage` with an affine and binning, `recenter`'s request and
  shifted pattern, `drift_correct` on shifted frames (both references),
  `auto_exposure`, `settings_schedule`, `game_of_life` read-back and a
  misregistered camera, `pyclm check` on `[pattern.inner]`, the pattern
  process building the inner method, `stage_xy`, the tracker keeping ids
  across a move, grid tiles moving, and the Controller's affine.
- The gallery built locally; the new cards looked at.
