# Changelog

## Unreleased

- **New pattern methods (2.0, second part of the method expansion; design
  note `docs/pattern-methods-expansion-design.md`).** Released as an
  experimental 2.0 pre-release.
  - Static: `rectangle` (`width_um`, `height_um`, offset `x_um` / `y_um`,
    `angle_deg`), `dose_grid` (rows × cols blocks or spots, one dose each),
    `image_mask` (a TIF projected centered or stretched, graded or
    thresholded).
  - Timed: `pulsed` (any method, the `[pattern.inner]` table, gated by a
    pulse train) and `chirp` (whole-field pulses whose period sweeps, log
    or linear).
  - Per cell: `all_cells` (`invert` lights the background, `dilate_um`),
    `random_cells` (a seeded fraction, fixed per track), `select_cells`
    (by area, intensity, distance from the center or the colony edge,
    `min` / `max` / `top_n`; `band_um` lights a band at the colony edge).
  - Per nucleus: `nucleus_clamp` gains `controller = "p"` / `"pi"` (`kp`,
    `ki`, `dose_mode`); `bang_bang` stays the default and unchanged.
    `population_clamp` holds the field's mean nuclear intensity with one
    input (a whole-field dose, the stimulation exposure, or a device
    property). `relative_clamp` calibrates each tracked cell's own dark
    and lit levels and holds it at a fraction of that range.
  - Adaptive (new zoo family): `recenter` and `drift_correct` move the
    stage; `auto_exposure` adjusts a channel's exposure; `settings_schedule`
    steps an exposure, preset or device property. Each wraps an optional
    `[pattern.inner]` method.
  - Diagnostic (new zoo family): `game_of_life`, a check of the DMD
    calibration on a fluorescent slide.
- Graded doses share one helper (`pyclm.core.patterns.dose`): `dose_mode =
  "gray"` (default; the DMD shows gray levels), `"dither"` (ordered dither)
  or `"duty"` (a fraction of every `period_min`).
- `WrappingPattern` (`pyclm.core.patterns.composite`) is the base of the
  methods that wrap one inner method; `pyclm check` validates
  `[pattern.inner]`, and `set_pattern` reaches it as `inner.<argument>`.
- Stage moves requested by a pattern method: `PatternMethod.camera_to_stage`
  converts an image offset to a stage move through MicroManager's
  pixel-size affine (`CameraProperties.camera_to_stage`; pixel size ×
  identity without one). On a grid, an x / y / z change moves every tile
  by the same offset (it used to change only the grid center, which the
  microscope ignores), and `pfs_offset` reaches every tile. Each
  acquisition event records the stage x / y it was made at
  (`stage_xy`), and the tracking process tells the tracker how far the
  objects jumped (`TrackingMethod.shift`; `centroid` shifts its
  centroids), so track ids survive a move.
- A method's requests for the same channel are merged into one
  (`merge_requests`), as `split` already did for its halves.
- The method zoo lists methods whose card waits for sample data
  (`ZooMeta.pending`) with their TOML, and shows the setting changes an
  adaptive example requests.

- **Pattern-method names and arguments follow one convention (2.0, first
  part of the method expansion; design note
  `docs/pattern-methods-expansion-design.md`).** A quantity with a unit
  names it (`radius_um`, `speed_um_per_min`, `width_um`, `spacing_um`,
  `speed_deg_per_min`), time is always in minutes, closed-loop methods
  need `channel` (no scope-specific default), and stripes are given as
  `width_um` + `spacing_um` instead of `duty_cycle` + `period`. Renamed
  methods: `direction` → `move_sequence` (`period_h` → `step_min`),
  `rotate_ccw` → `move_rotate` (+ `reverse`), `rotate_bar` → `spokes`
  (`speed_deg_per_min`, positive counterclockwise), `binary_nucleus_clamp`
  → `nucleus_clamp` (`clamp_target` → `target`), `centered_image` →
  `target_image` (`path`, `target_min`, `target_max`). Renamed arguments:
  `circle` `rad` → `radius_um`; the bar family and `rings` `bar_speed` /
  `wave_speed` → `speed_um_per_min`, `t_loop` → `loop_min`, `inverse` →
  `reverse`, `rings` `direction = -1` → a negative speed; `blinking_bar`
  `duty_cycle` → `width_um`; `global_pulse` `reference_bar_speed` →
  `reference_speed_um_per_min`; embryo `fraction_length` → `fraction`;
  `nuc_channel` → `channel` (it used to be silently ignored). The unused
  integer `direction` of the per-cell methods is gone.
- Old names and arguments keep working: the method translates them
  (`PatternMethod.migrate_kwargs`, `renamed_kwargs`, `legacy_defaults`;
  `renamed_method` makes an alias) wherever it is built, with one warning
  per run, and `pyclm check` warns with the replacement. `pyclm check
  --fix <dir>` rewrites the experiment TOMLs (comments kept, originals
  saved as `.toml.bak`). All 22 old-style configurations compared give
  pixel-identical patterns to before.
- Every pattern method takes `start_min` / `stop_min` and is dark outside
  that window (`PatternMethod.render`, which the pattern process,
  `preview`, `split` and the zoo call; methods still implement
  `generate`).
- `pyclm check` accepts the arguments a method passes on to its parents
  (`start_min`, the per-cell `voronoi` / `gradient` / `tracks`), which it
  used to report as unknown.
- Fixed: `embryo_inner` and `embryo_outer` raised a broadcasting error on a
  non-square field (`np.meshgrid` without `indexing="ij"`).
- The method zoo opens with a grid of thumbnail cards, one per built-in
  pattern method, grouped by family, animated for methods that change over
  time, tagged with what each needs (open loop, segmentation, tracks), and
  filterable by family or loop. Each links to an entry with a `[pattern]`
  table to paste, the sample image, the pattern the DMD receives, and a
  filmstrip or a timeline. Every registered method is now covered (9 were
  missing). A test fails for a built-in without `zoo_meta`. Design note:
  `docs/method-zoo-design.md`.
- `method = "rings"` (concentric rings moving out or in; `wave_speed = 0`
  holds them still) is registered. It had a zoo card as "wave", but no
  TOML name.
- **Behaviour change:** `direction`'s `"u"` and `"d"` were swapped: `"u"`
  lit the lower side of each cell. Cells move towards the light, so `"u"`
  now lights the upper side (towards row 0) and `"d"` the lower. An
  experiment written against the old letters should swap them.
- `global_cycle`, `move_down` and `fb_bounce` are deprecated: they are
  fixed cases of `global_pulse` and `direction`, and `pyclm check` warns
  with the equivalent arguments. They still run.
- The per-cell methods with `gradient = true` no longer put NaN in the
  pattern for a one-pixel-wide cell (known issue 37). The zoo examples of
  `move_in`, `move_out`, `rotate_ccw` and `split` use `gradient = true`.
- `fb_bounce` reverses every half-loop; it reversed once and then kept
  one direction for the rest of the run (known issue 36).
- `pyclm preview --t START:STOP[:STEP]` previews a stretch of the schedule
  in seconds: the method runs at each timepoint with its state kept, and
  the preview adds pattern stacks, a captioned montage and an animation.
- Agent skills for new users in `.claude/skills/`: microscope bring-up,
  experiment design, and writing a pattern method. The README points to
  them. `claude.md` is now `CLAUDE.md`, so Claude Code loads it on
  case-sensitive file systems. The first-time setup now gives the position
  naming rule the code uses (`<toml stem>.<suffix>`; it said `-`).

## 1.1.0 — 2026-10-01 (core refactor)

This release rebuilds the internals of PyCLM in four staged, individually
shippable steps while keeping the experiment TOML format, the
`generate(context)` / `segment(img)` method signatures and every existing
output file readable. It adds a declarative acquisition plan, a new storage
format that standard tools can open, a router that makes new processing
stages a registration rather than a surgery, live tracking, several
segmentations per channel, a measurement toolbox for pattern methods, and
documented, tested examples of advanced closed-loop experiments.

### How this release was made

Previous versions of PyCLM were designed and written by hand by the lab.
The work in this release was implemented primarily with **Claude Code**
(Anthropic's Claude, model Fable 5.1) working in this repository, and the
split of responsibilities is worth recording:

- **Human (Harrison Oatman, PyCLM's designer).** Set the goals (the five
  planned features: tracking, runtime edits, z-stacks, grids, interactive
  setup), read and approved each stage's design document before any code
  was written, took every recorded design decision (for example: keep
  today's cadence semantics; one OME-Zarr store per experiment; cadence
  groups instead of blank frames; save DMD patterns on change; the router
  as an in-thread publisher; the activation rule for processing stages;
  the built-in tracker; flipping the default storage format), raised the
  design questions that produced cadence groups, the pattern save-on-change
  policy, the measurement toolbox and named segmentations, tested Stage 2
  on the microscope, and made the commits for Stages 0–2.
- **Claude Code.** Wrote the architecture assessment and the per-stage
  design documents in `docs/`, the code, the tests (8 at the start, 170
  now), the user documentation, and this changelog, and ran the dry-run
  and unit tests after each change. It did not run on the microscope.

Every design document in `docs/` lists what was verified empirically and
how, so a reader can check the claims rather than trust them. Code review
by a second human has not yet happened and is recommended before this
becomes a tagged release.

### Stage 0 — hygiene

- Inter-process queues are `queue.Queue` (threads share objects; no copies).
- The Manager's wait loop sleeps instead of spinning.
- **Bug fix:** with `t_delay > 0`, pattern requests used the
  experiment-relative timepoint while frames used the absolute one, so
  closed-loop patterns were never generated. Both now use the absolute
  timepoint.
- Output files are closed on forced shutdown; every write is flushed.
- Per-run logging (`log.log` per experiment directory); `print` replaced
  by module loggers except for the operator progress line.
- The microscope thread has a per-message exception guard and aborts only
  after ten consecutive failures; the PFS focus wait cannot hang.
- `settle_time_seconds` and `focus_device` are configurable in
  `pyclm_config.toml`.
- Method registries are per controller instance; `pattern_shape` stays
  integral under binning; `PatternReview` is constructible from a TOML;
  pre-existing outputs are detected before any model loads; the TOML
  parser no longer mutates its input; lint debt cleared (ruff via
  pre-commit).

### Stage 1 — the acquisition plan

- The schedule is data: `AcquisitionPlan` wraps a useq-schema
  `MDASequence` (time plan, positions, channels, exposures, PFS offsets)
  with PyCLM's cadence rules (`every_t` counted from `t_delay`, `t_stop`,
  per-position offsets, the stimulation channel and its SLM handshake).
- One `is_scheduled` instead of four re-derivations; the Manager walks
  `plan.events_at(t)`; the writer and GUI derive their layout from the plan.
- Every frame has a stable identity `index = {"t", "p", "c"}` (replacing
  `sub_axes`), ready for z and grid axes.
- `plan.useq.yaml` is written next to the data and embedded in every output.
- A timing budget (`plan.over_budget`) warns at start-up when a timepoint
  cannot fit in the interval.
- The stimulation channel is named by its preset in the channel group;
  the PFS offset is recorded in the plan for provenance.
- New dependency: `useq-schema`.

### Stage 2 — storage

- `FrameWriter` interface behind the writer; the original HDF5 layout is
  preserved as format 1 (`HDF5WriterV1`).
- **OME-Zarr (NGFF 0.4 on zarr v2) is the new format 2 and, since
  Stage 3, the default.** One store per experiment; channels sharing a
  cadence form one image with a compact time axis (no blank frames for
  skipped timepoints); zstd-compressed chunk per frame; skipped frames
  cost nothing on disk; opens in Fiji, napari, QuPath and plain Python.
- DMD patterns are stored once per distinct pattern (`pattern_policy =
  on_change`, also `all` / `none`); a `frames.parquet` / `frames.csv`
  table records every frame and stimulation event with timestamps,
  position, exposure, binning and the pattern in force.
- `pyclm.io` reads both formats behind one API (`open`, `ExperimentData`,
  `GroupData`, `find_experiments`); the GUI and `convert_hdf5s` use it.
- ImageJ hyperstack export (raw, labels, tracked labels, pattern overlay;
  calibrated frame interval and pixel size), automatic at the end of a
  run (`[output] export_imagej`).
- New `[output]` section in `pyclm_config.toml`: `format`,
  `pattern_policy`, `export_imagej`.
- New dependencies: `zarr >= 3`, `pyarrow`.

### Stage 3 — router, registration, history, tracking

- **Router.** Frame-derived data (`raw`, `seg`, `seg:<name>`, `tracks`)
  is fanned out by a subscription table built once from what each
  process declares it wants and produces. Segmentation and tracking run
  only where a consumer demands their output, at the cadence it needs.
  Shutdown fan-in is derived from the same table; no process counts its
  upstreams. The resolved table is stored in every output for provenance.
- **Registration.** `PipelineProcess` and `Controller.add_process()`;
  `MicroscopeOutbox` is now `WriterProcess` (the old name is an alias).
- **Pattern history.** `add_requirement(..., history=n)`;
  `context.history()`, `context.last_pattern()`,
  `context.pattern_history()`, `context.t`, `context.generation`.
- **Tracking.** `TrackingMethod` plug-in surface, the built-in `centroid`
  linker, `TrackingProcess`, `[tracking]` in the experiment TOML,
  `context.tracks(channel)` (a `Tracks` object), tracked labels and a
  `tracks.parquet` table in the store, read by `pyclm.io`, exported to
  ImageJ. The per-cell pattern bases (`rotate_ccw`, `move_out`,
  `binary_nucleus_clamp`, ...) accept `tracks = true` to run on tracked
  labels.
- **Named segmentations.** `[segmentation.<name>]` tables give an
  experiment several segmentations of the same frames, each with its own
  method; `add_requirement(channel, seg="nuclei")`; one label image per
  table; `[tracking] segmentation = "<name>"`.
- **Measurement toolbox** (`pyclm.Regions`, `pyclm.PerTrack`,
  `pyclm.nuclear_cytosolic_ratio`; `context.regions()`): per-object
  measuring, per-track memory and painting per-object values back into a
  pattern, so a per-cell controller is a few lines.
- **Documented, tested examples**: leader cells with directed
  stimulation, a three-phase per-cell intensity programme, a KTR
  nuclear/cytosolic ratio clamp (`documentation/examples/`).

### Stage 4 — runtime setting changes and feedback from the microscope

- **A pattern method can change settings of its own experiment** while
  the run is in progress: `context.set_exposure`, `set_config`,
  `set_property`, `set_position`, with `context.settings(channel)` and
  `context.position()` for read-back. Changes apply from the next
  timepoint (the same delay as the pattern); the schedule, channels and
  cadence never change. The driving case is a red / far-red optogenetic
  tool switched on and off from a programme string through the lasers'
  device properties (documented, tested example).
- **Provenance.** `events.parquet` / `events.csv` record every request
  (requested and applied timepoints, old and new value, applied or
  refused), focus-lock corrections, late timepoints and acquisition
  errors; `pyclm.io` exposes it as `ExperimentData.events`. The frames
  table gains one column per device property or config group a method
  changed, holding the value in force on every later frame.
- **Acknowledgements and status.** The microscope acknowledges every
  acquisition; the Manager tracks lateness and errors, warns when a
  timepoint finishes more than one interval late, and writes
  `status.json` every timepoint (progress, per-experiment lateness and
  errors, applied and refused settings, process health). The live GUI
  shows a status line.
- Deferred from the original Stage 4 plan: pause/resume, extending a run,
  hot-reload of the experiment directory, a command file or socket, GUI
  controls, skipping late timepoints, multiple DMDs.

### Stage 5 — configuration schema, `pyclm check`, one command, preview

- **A schema for the three kinds of file** (`pyclm/schema.py`, pydantic,
  already a dependency): every key has a type, a default and a limit;
  unknown keys are errors with a did-you-mean; every problem in a file is
  reported together in the file's own vocabulary
  (`bar10.toml: unknown key 'exposur' in [imaging] (did you mean 'exposure'?)`).
  `format_version` keys are accepted. The TOML reference in the docs is
  generated from the schema.
- **`pyclm check <dir>`**: files, positions against experiment files,
  method names and arguments against constructor signatures, what the
  pattern method asks for against the tables, presets and devices against
  the MicroManager `.cfg` (parsed as text), the timing budget, existing
  outputs; `--dry` rehearses two timepoints on the virtual microscope.
  `pyclm run` performs the same check and refuses to start on errors.
- **One command**: `pyclm new | check | preview | run | export | gui`
  (`pyclm <dir>` still means `run`; `convert_hdf5s` and `gui` remain).
- **`pyclm preview`**: the experiment's segmentation(s) and pattern method
  on one TIF or on frames snapped from the microscope, writing the raw
  image, labels, the pattern in camera and DMD space, an overlay, and a
  JSON summary, through the run's own machinery.
- **`pyclm new`**: an open-loop or closed-loop directory from the lab's
  experiments, with a README of next steps.
- Two long-standing configuration bugs found by the schema: per-channel
  `config_groups` / `device_properties` overrides were never applied, and
  `t_delay` / `t_stop` written after `[pattern]` (as the docs showed) were
  silently passed to the pattern method instead of delaying the experiment.

### Stage 5b — commands during a run, the viewer's minimap and positions axis

- **The viewer stays on napari** (assessed against ndv and pymmcore-gui:
  ndv is alpha without label layers, pymmcore-gui pins PyQt6 and zarr < 3)
  and gains a dock with a status line that updates every second and a
  **minimap** of the positions in stage coordinates, coloured by
  experiment, the current position filled, fields of view drawn.
- **Positions are an axis in the viewer**, not layers: one layer per
  channel stacked over (position, t, y, x), so contrast is one setting per
  channel for every position and the layer list stays short. A positions
  list in the dock, the minimap, the position slider and the `[` / `]`
  keys switch position; **Follow the run** tracks the acquisition.
  Auto-contrast stops once you move a slider (before, every new frame reset
  it) and can be turned back on or applied once.
- **Commands to a running experiment**: one JSON file each in
  `commands/`, applied at the timepoint boundary, recorded in the events
  table, moved to `commands/done/`: `pause` / `resume` (the clock shifts by
  the paused time), `stop_run`, `stop_experiment`, `set_exposure`,
  `set_config`, `set_property`, `set_position`, `set_pattern` (a new
  `PatternMethod.update(**parameters)` hook). `pyclm.commands.write_command`
  writes one from Python.
- **A control window (`pyclm control`) was built and withdrawn** the same
  day, at Harrison's request, before it had been used at the microscope:
  a run panel driving `pyclm run` as a subprocess with the commands above,
  a positions table fed by the stage with a `PositionList.pos` writer, and
  configuration forms generated from the schema (pymmcore-widgets panes
  for stage, snap and live were part of it). It lives in the git history
  of `gui/control.py`, `gui/forms.py` and `tests/test_control.py` should
  it come back. `directories.write_position_list` stays.
- **Dependencies**: useq-schema ≥ 0.9.2, napari ≥ 0.9. pymmcore / pymmcore-plus ranges are open
  (≥ 11.2.1.71.0 / ≥ 0.14.0) and the lock is constrained to the last
  **Micro-Manager device interface 71** stack (pymmcore 11.2.1.71.0,
  pymmcore-plus 0.14.0) because the lab's Mightex
  Polygon adapter is a closed DLL built for interface 71. The suite passes
  on that stack and on the current interface-75 stack (pymmcore 12.5,
  pymmcore-plus 0.18.1); `pyclm check` prints the
  interface in use and a mismatched configuration load says why it failed.

### Dry run: pixel size and binning of the TIFs

- `dry_run.yml` accepts `pixel_size_um` and `binning` (top level, optional,
  also without `positions`). The virtual microscope reports the pixel size,
  and the camera-to-DMD affine is scaled by the binning, so 4x-binned test
  images map onto the DMD the way the unbinned camera does. `pyclm preview
  --image` reads the same keys when `--pixel-size-um` is not given and
  reports `source_binning`. Before this, the dry run assumed 0.33 µm and
  unbinned frames, and binned images lit almost nothing on the DMD.

### Stimulation-only experiments

- `[channels]` may be omitted or have `presets = []`: the experiment only
  stimulates. The schema no longer requires at least one preset (a
  constraint Stage 5 added without a runtime reason; the plan, storage and
  viewer already handled a stimulation-only group), `pyclm check` reports
  it, and `pyclm preview` probes with the stimulation frame.

### Stage 6 — grids, skipping idle positions, the camera ROI, patterns in both spaces

- **Grid acquisition** from MicroManager's Create Grid: the tiles of a grid
  (labelled `<prefix>-<n>-<col>_<row>`, with `GridRow` / `GridCol`) fold
  into one position at their centre. Every channel visits all the tiles at
  each timepoint (through the configured position mover, so PFS locks per
  tile) and publishes one stitched frame; the pattern method's
  `pattern_shape` is the stitched frame and its pattern is cut into one DMD
  image per tile. Stitching is placement by row and column (MicroManager's
  pixel-size affine makes tiles abut), no registration. `pyclm check`
  describes each grid and, with `camera_roi` and `--pixel-size-um`, whether
  the tiles abut. Grids need OME-Zarr. `context.grid()` for methods.
- **Skipping idle positions.** A position with nothing due was already
  skipped; now a position whose only due acquisition is the stimulation
  frame is skipped when the pattern the SLM handshake delivers is blank and
  the method does not need that frame: no stage move, no exposure. The
  handshake now precedes the stage move (per position: request pattern,
  update pattern, position, acquisitions). Skipped frames are rows of
  `kind = "skipped"`, `position_skipped` events, `skipped` in `status.json`.
- **Camera ROI**: `camera_roi = [x, y, w, h]` in `pyclm_config.toml`, set
  at run start and by preview; the affine stays in full-frame coordinates
  and PyCLM composes the offset. `pyclm check` prints the DMD's footprint
  on the camera and warns when the ROI exceeds it.
- **Patterns in both spaces** (OME-Zarr format 3): `patterns/camera/0`,
  what the method returned (stimulation binning, ROI frame, stitched for
  grids), alongside `patterns/dmd/0`; `pattern_index` now points into the
  camera array and the new `dmd_index` into the DMD array (per-tile entries
  for grids, ids and tiles in the attrs). `pattern_policy` gains `imaging`
  (store only at timepoints with a saved imaging frame) and loses `all`.
  HDF5 format 1 keeps its layout and its `dmd` datasets are now gzip
  compressed, which is where most of a long run's bytes went. The viewer
  and the export overlay the camera-space pattern directly.
- **Removed**: the `pattern_review` method (`PatternReview`), and with it
  `PatternMethodReturnsSLM` and the SLM-coordinates return path; every
  method returns camera coordinates.
- Z-stacks were designed (a per-channel projection to the pipeline, the
  stack beside it) and dropped before implementation: the geometry would
  have needed a second MicroManager export. The design is in the Stage 6
  notes should it return.

### The position mover in `pyclm_config.toml`

- `position_mover = "basic" | "pfs" | "package.module:ClassName"` chooses
  how the stage moves to a position. Until now only `run_pyclm(...,
  position_mover=...)` in code could choose, so every command-line run used
  the basic mover and a Nikon's PFS was never engaged. `pyclm check`
  reports the resolved mover; a mover passed in code still wins.

### Labelled schedule and configuration files

- `schedule.<anything>.toml` and `pyclm_config.<anything>.toml` are
  recognised as the schedule and the configuration (`schedule.toml`,
  `pyclm_config.toml` still work); every other TOML is an experiment. Two
  schedules or two configurations in one directory are an error, in
  `pyclm check` and at run start.

### Documentation

- The API reference is reorganised by task (running and preparing,
  configuration, writing methods, reading data, the shipped pattern
  methods, the pipeline for developers) and covers the modules added
  since Stage 3; pages are named after their modules.
- `docs/release-wet-tests.md` lists the checks that need a microscope
  before a release, and `tests/wet_tests/` is a self-contained package
  that runs them: one experiment directory per item with a `guide.md`
  saying what must be generated live, a `wet` command (`prepare`, `run`,
  `verify`, `commands`, `report`, `clean`) that copies one configuration
  template into every item and registers the methods the items use, and
  automatic verification of the files a run leaves behind.
- A dry run no longer takes a previous run's ImageJ hyperstacks
  (`<label>_imaging.tif`, `<label>_stim.tif`, …) for position TIFs.

### The `split` pattern method

- `method = "split"` with `[pattern.left]` / `[pattern.right]` (or `top` /
  `bottom`) sub-tables runs one pattern method on each half of the field:
  each half is an ordinary method with its own arguments, requirements are
  merged, `boundary` moves the edge and `feather` blends across it,
  `pyclm check` validates the halves, `set_pattern` reaches them as
  `left.<argument>`. Two hooks on `PatternMethod` make it possible for
  any composite: `nested_methods(kwargs)` (what the check validates) and
  `bind_registry(registry)` (called by the pattern process before
  `initialize`, with the registered methods).

### The pixel size follows the objective

- The run applies the config-group presets every experiment shares (the
  objective among them) before reading the camera's pixel size, so the
  pixel size no longer depends on which objective the microscope was left
  at; a pixel size of 0 is warned about. `pyclm check` reads the `.cfg`'s
  pixel-size presets, prints the one each experiment's `[config_groups]`
  select, warns when none matches, and warns when experiments in a
  directory disagree on a global config group (the affine and camera ROI
  are per objective).

### Fixes

- A negative `bar_speed` moves the bar (and the sawtooth) towards -y; it
  used to make the modulo negative and light the field wrongly. The period
  in time now follows a `set_pattern` change of `bar_speed`.
- OME-Zarr metadata writes (progress, pattern ids, array resizes) and
  the parquet tables retry on the Windows sharing violation a reader can
  cause (known issue #33).
- The PFS mover switches focus maintenance off and on again (every 5 s
  while waiting for a lock) when the PFS has switched itself off; a
  repeated "On" alone was ignored and every later position timed out
  (known issue #34). The timeout message lists the status strings seen.

### Distribution: `closed-loop-microscopy` on PyPI, custom methods from the command line

- The package is published as **`closed-loop-microscopy`** (the import
  name and the command stay `pyclm`): `pip install
  "closed-loop-microscopy[gui]"` on a microscope PC, without `[gui]` for a
  headless install (check, preview, dry runs, reading data). Version
  1.1.0; `pyclm.__version__` is recorded in the OME-Zarr attributes and the
  plan metadata.
- The viewer is the **`gui` extra** (napari, PySide6). `pyclm gui` and
  `run --gui` say how to install it when it is missing, before the
  hardware is touched. In a clone, `uv sync` still installs it (the default
  `dev` group includes the test group, which carries it).
- Core dependencies trimmed: `opencv-python-headless` instead of
  `opencv-python` (no second Qt next to napari's), `pydantic` and `scipy`
  declared, `toml` replaced by the standard library's `tomllib`,
  `notebook` moved to the `analysis` group and `pandas-stubs` to `dev`.
- **`methods` in `pyclm_config.toml`**: Python files (relative to the
  configuration) or module names whose `PatternMethod` /
  `SegmentationMethod` / `TrackingMethod` subclasses are registered by
  their `name`, for `pyclm run`, `check` and `preview`; installed packages
  register theirs with a `pyclm.methods` entry point (`pyclm/methods.py`).
  `pyclm check` lists what it loaded and reports a file that does not
  import, a class without its own `name`, a built-in name, or one name in
  two places; an unknown method's error says where custom methods are
  listed. Methods passed to `run_pyclm` in code still work and win.
- **`pyclm new <dir> --demo`**: two experiments (an open-loop bar; a
  closed loop with a threshold segmentation defined in the directory's
  `demo_methods.py` and loaded through `methods`), synthetic cell images
  drawn on the spot, a `dry_run.yml` and a calibration covering the whole
  simulated camera, so `pyclm run <dir> --dry` works on any computer.
- **`CITATION.cff`**: the software's authors (Oatman, Gubbens, Toettcher)
  and the Cell Systems paper as the preferred citation; the README has a
  citation section.
- **Release checks**: `scripts/check_release.py <tag>` fails unless the
  tag is `v` + the `pyproject.toml` version and `CHANGELOG.md` has a dated
  section for it (and `CITATION.cff` the same version); the publish workflow runs it and the test suite before
  building.
- TOML files are read with `tomllib` (TOML 1.0); a file the old `toml`
  package accepted but that is not valid TOML 1.0 is now an error that
  `pyclm check` names.

### Breaking changes for developers

- `AcquisitionEvent` lost its routing arguments (`do_segmentation`,
  `save_segmentation`, `raw_goes_to_pattern`,
  `segmentation_goes_to_pattern`, `segmentation_method`,
  `pattern_method`, `save_pattern`, `save_stim`) and `sub_axes` (use
  `index`). `PlannedEvent` lost the same flags.
- `AllQueues` keeps only control queues; data edges are the router's.
- `SegmentationProcess.models` is keyed by `(experiment, segmentation
  name)`; `request_method(experiment, name)`.
- `Experiment.segmentation` is a property over `Experiment.segmentations`;
  `TrackingConfig` is a class with a `segmentation` attribute.
- `AcquiredImageRequest` gained `needs_tracks`, `history` and
  `segmentations`; use `.kinds` rather than testing the booleans.
- `PatternContext` is built on an `ExperimentState`; `DataDock.add(data)`
  slots by `data.kind`.
- `GroupData.labels(i, channel, name="segmentation")`; `has_labels` is
  derived from `label_names`.
- The default `[output] format` is `ome-zarr`; set `hdf5` to keep the old
  layout. HDF5 format 1 does not store tracks or named segmentations.
- `Manager.initialize(plan, event_log=, status_path=, health=)`; the
  microscope now sends `EventDoneMessage` after every acquisition on
  `microscope_to_manager`; `AllQueues` gained `pattern_to_manager`;
  `AcquisitionEvent.overrides` carries runtime-changed settings.
- `t_delay` semantics were fixed rather than preserved (no experiment
  had used it).
- Configuration files: unknown keys are errors; `steps` and
  `interval_seconds` are required in `schedule.toml`; `t_delay` / `t_stop`
  must precede the first table. `experiment_from_toml`, `read_schedule`
  and `run_pyclm` raise `pyclm.schema.ConfigError` (a `ValueError`) instead
  of `KeyError`. `run_pyclm(..., check=True, force=False)` runs the check
  first and raises `pyclm.check.CheckFailed` on errors.
- The `pyclm` entry point is `pyclm.cli.main` with subcommands.
- OME-Zarr stores are format 3: `pattern_index` indexes `patterns/camera/0`
  (it indexed `patterns/dmd/0` in format 2; `pyclm.io` reads both);
  `pattern_policy = "all"` is gone (`on_change`, `imaging`, `none`).
- The Manager emits `update_pattern` before `position` at each position;
  `PlannedEvent` and the three control events carry `skippable`;
  `EventDoneMessage.skipped`; a `skipped` routing kind (`SkippedAcquisition`)
  that the writers subscribe to for the stimulation channel;
  `FrameWriter.write_skipped`.
- `MicroscopePosition` gained `tiles`, `grid`, `grid_rc`, `geometry`;
  `positions_from_pos` folds tile-creator entries; `image_shape(core,
  binning, geometry)`; `SLMBuffer.initialize(..., roi, grids)`;
  `EventSLMPattern` and `StimulationData` carry `camera_pattern` and
  `dmd_ids`, and a grid's DMD pattern is a list.
- `PatternReview` / `PatternMethodReturnsSLM` / `CameraPattern.slm_coords`
  removed.
- pymmcore and the Micro-Manager device adapters must share a device
  interface. `uv.lock` is held at interface 71 by `[tool.uv]
  constraint-dependencies` in `pyproject.toml`; delete them and `uv lock`
  to move to the current stack. Do not pair pymmcore-plus 0.15.0 with
  pymmcore 11.2.1: it fails on import.
- `Manager.initialize(..., commands_dir=)`; `status.json` gained
  `current_experiment`, `paused`, `paused_s`, `stopping`,
  `pending_commands`, `commands_applied`; `events.parquet` gained a
  `source` column; `AcquisitionPlan.stop_experiment`; cores gained
  `getXYPosition`.

### Not changed

Experiment TOML keys (only added: `[tracking]`, `[segmentation.<name>]`,
`[output]`), `PatternMethod.generate(context)`,
`SegmentationMethod.segment`, `PositionMover.move_to`, the virtual
microscope and dry runs, and readability of every HDF5 file written by
earlier versions.

### Still to come (see `docs/assessment-2026-09.md`)

Later: z-stacks and grid acquisition on top of the plan; a LapTrack
tracking method; a single window if ndv gains label layers.
