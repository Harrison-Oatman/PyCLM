# Changelog

## Unreleased — next major release (core refactor, Stages 0–3)

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

### Not changed

Experiment TOML keys (only added: `[tracking]`, `[segmentation.<name>]`,
`[output]`), `PatternMethod.generate(context)`,
`SegmentationMethod.segment`, `PositionMover.move_to`, the virtual
microscope and dry runs, and readability of every HDF5 file written by
earlier versions.

### Still to come (see `docs/assessment-2026-09.md`)

Stage 5b: interactive setup (forms from the schema, a positions editor,
run control in the GUI) once the command line has been used by a new lab
member. Later: pause/resume and operator commands on the same boundary
Stage 4 established, z-stacks and grid acquisition on top of the plan.
