@ -1,79 +0,0 @@
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for package management.

```bash
# Run tests
uv run --group test pytest

# Run a single test
uv run --group test pytest tests/test_base_process.py

# Lint (via pre-commit)
uv run pre-commit run --all-files

# The pyclm command (see documentation/command_line.md)
uv run pyclm new <dir> [--template open-loop|closed-loop] [--name NAME]
uv run pyclm check <dir> [--config path/to/pyclm_config.toml] [--mm-config file.cfg] [--dry]
uv run pyclm preview <dir> <experiment> --image frame.tif
uv run pyclm run <dir> [--config path/to/pyclm_config.toml] [--dry] [--gui] [--force]
uv run pyclm export <dir> [channels]      # ImageJ hyperstacks (convert_hdf5s is an alias)
uv run pyclm gui <dir>                    # the live viewer (napari; status line + minimap dock)
uv run pyclm control <dir> [--dry]        # the control window: run / commands, positions, files
```

**Device interface:** pymmcore must match the Micro-Manager adapters' device interface; the lock is held at interface 71 (Mightex Polygon DLL) by `[tool.uv] constraint-dependencies` in `pyproject.toml`, open ranges otherwise. See `docs/stage5b-interactivity-design.md` "Device interface" before touching pymmcore pins.

**Linting:** ruff (formatter + linter with rules B, I, RUF, PT, UP) and nbstripout for notebooks — configured in `.pre-commit-config.yaml`.

**Optional dependency groups:**
- `cellpose` — CellposeSAM segmentation (requires torch/CUDA)
- `calibration` — DMD calibration via pycromanager
- `analysis` — data analysis (matplotlib, pandas, laptrack, seaborn, etc.)
- `dev` — analysis + docs + pytest

## Architecture

PyCLM is a closed-loop microscopy system that runs multiple simultaneous optogenetic experiments. The core architecture is a multi-threaded pipeline of processes: control messages travel on queues, frame-derived data is fanned out by a router built from the methods' declared requirements. Engineering notes (runtime map, known issues, per-stage design records) live in `docs/`; start with `docs/README.md`.

### Process Pipeline

`Controller` (`controller.py`) owns all processes and runs them in a `ThreadPoolExecutor`. The processes are:

1. **`Manager`** (`core/manager.py`) — Timing brain. Walks the `AcquisitionPlan` (`core/plan.py`, a useq `MDASequence` plus PyCLM cadence rules) and turns each timepoint into messages for the microscope, SLM buffer and pattern process. Between timepoints it applies the setting changes pattern methods requested for their own experiment (`core/settings.py`: exposure, presets, device properties, position; the schedule itself never changes), polls `commands/` for operator commands (`commands.py`: pause/resume, stop, set_*, set_pattern), absorbs the microscope's acknowledgements, and writes `status.json` and `events.parquet`. First to finish in a normal run; its exit triggers graceful shutdown.
2. **`MicroscopeProcess`** (`core/microscope.py`) — Controls hardware via pymmcore-plus. Executes acquisition events, updates stage positions, applies SLM patterns, publishes every frame to the router.
3. **`WriterProcess`** (`core/writer_process.py`) — Hands frames, label images and track tables to the configured `FrameWriter` (`core/storage/`: OME-Zarr format 2 by default, HDF5 format 1).
4. **`SLMBuffer`** (`core/manager.py`) — Holds the current DMD pattern per experiment, applies the affine transform from camera to SLM coordinates, sends patterns to the microscope on demand.
5. **`SegmentationProcess`** (`core/segmentation_process.py`) — Runs one segmentation method per configured `[segmentation]` table on the frames the router delivers, each at the cadence its consumers need.
6. **`TrackingProcess`** (`core/tracking_process.py`) — Runs a `TrackingMethod` on every segmentation of a channel a pattern method wants tracks of.
7. **`PatternProcess`** (`core/pattern_process.py`) — Generates light patterns using registered `PatternMethod` instances, with a bounded per-experiment history.

Control messages use the queues in `core/queues.py` (`AllQueues`). Frame-derived data (`raw`, `seg`, `tracks`) goes through the `Router` (`core/router.py`): at `Controller.initialize` it resolves a subscription table from what each `PipelineProcess` (`core/base_process.py`) declares it wants and produces, fans data out with `publish`, and derives the shutdown stream-close from the same table. Segmentation and tracking run only where some consumer demands their output. Extra consumers/producers are registered with `Controller.add_process()`.

### Experiment Configuration

Experiments are configured entirely via TOML files in an experiment directory:

- **`[experiment].toml`** — One per experiment. Defines imaging channels, stimulation, segmentation methods (`[segmentation]` plus named `[segmentation.<name>]` tables), tracking method, and pattern method with kwargs.
- **`schedule.toml`** — Timing: `[timing]` section with `steps`, `interval_seconds`, `setup_time_seconds`, `time_between_positions`.
- **`multipoints.xml`** — Imaging positions exported from MicroManager's multipoint list. Position labels link to experiment TOMLs (e.g., position `feedbackexp.1` uses `feedbackexp.toml`).
- **`pyclm_config.toml`** — Hardware config: `config_path` (MicroManager .cfg), `affine_transform` (2×3 matrix, camera→SLM), `slm_shape_h`/`slm_shape_w`, optional `focus_device`, `settle_time_seconds`, and `[output]` (`format`, `pattern_policy`, `export_imagej`). Located at repo root or in the experiment directory.

`schema.py` holds the pydantic models for the three kinds of file (`ExperimentConfig`, `ScheduleConfig`, `PyclmConfig`; unknown keys are errors, method-table extras are the method's kwargs, `format_version`); `directories.py:schedule_from_directory()` builds an `ExperimentSchedule` through them. `check.py` (`pyclm check`, also run by `pyclm run`) validates a directory: files, positions ↔ TOMLs, method names and arguments against constructor signatures, requirements against tables, presets against the MicroManager `.cfg` (`mmconfig.py`, text parsing), the timing budget, existing outputs. `preview.py` runs one experiment's methods on one image; `templates.py` backs `pyclm new`; `cli.py` is the command. GUI: `gui/gui_controller.py` (napari viewer, read-only, own process; positions are a dims axis: one `ChannelStack` layer per cadence group / channel over (position, t, y, x), a `ViewerControls` positions list, follow-the-run, auto-contrast that stops when the user moves a slider), `gui/widgets.py` (status line, minimap; plain Qt, shared), `gui/control.py` (`pyclm control`: run panel driving `pyclm run` as a subprocess and writing command files, positions panel on `MMCoreStage` / `SimulatedStage` with pymmcore-widgets, forms), `gui/forms.py` (schema-generated forms saved with tomlkit). The GUI processes talk to a run only through files; the run has no Qt.

### Extending PyCLM

**Custom pattern methods:** Subclass `PatternMethod` (`core/patterns/pattern.py`). Implement `generate(self, context: PatternContext) -> np.ndarray` returning a float array (0–1) in camera coordinates. Use `add_requirement(channel_name, raw=, seg=, tracks=, history=)` in `__init__` to declare what data the method needs (`seg` is `True`, the name of a `[segmentation.<name>]` table, or a list of names). The measurement toolbox (`core/measure.py`: `Regions`, `PerTrack`, `nuclear_cytosolic_ratio`; `Tracks` is a `Regions`) does per-object measuring, per-track memory and painting. Register with `Controller.register_pattern_method(name, cls)` before `c.initialize()`.

**Custom segmentation methods:** Subclass `SegmentationMethod` (`core/segmentation/segmentation.py`). Register with `Controller.register_segmentation_method(name, cls)`. Several segmentations of one channel are configured as named `[segmentation.<name>]` tables and travel as kind `seg:<name>` (`core/kinds.py`).

**Custom tracking methods:** Subclass `TrackingMethod` (`core/tracking/tracking.py`, `track(labels, t, pixel_size_um) -> (relabelled, rows)`). Register with `Controller.register_tracking_method(name, cls)`; enable with a `[tracking]` table in the experiment TOML.

**Virtual microscope (dry run):** `--dry` flag activates `SimulatedMicroscopeCore` + `TimeSeriesImageSource` (TIFs in the experiment directory, mapped by `dry_run.yml`, a position list, or TIF names; `dry_run.yml` may also set `pixel_size_um` and `binning` of the TIFs, and the Controller scales the affine by that binning). Useful for testing pattern logic without hardware.

### Data Output

Each experiment produces one output: by default an OME-Zarr store (`<experiment>.zarr/`, one NGFF image per cadence group with a compact time axis, segmentations (one label image per `[segmentation]` table) and tracks as label images, DMD patterns stored once per distinct pattern) plus `frames.parquet` / `tracks.parquet` tables in the experiment directory; with `[output] format = "hdf5"`, the original per-timepoint HDF5 layout (`00001/channel_GFP/data`, `stim_aq/dmd`, SWMR). `pyclm.io` reads both; ImageJ hyperstacks are exported at the end of a run. See `documentation/data_format.md`.

### Key Types

- `AcquisitionEvent` — encodes a single image acquisition: identity from the plan (`index` = `{t, p, c}`), position, channel, exposure, `needs_slm`, `save_output`. Nothing about routing.
- `PatternContext` — passed to `PatternMethod.generate()`, provides `.raw(channel)`, `.segmentation(channel, name=)`, `.regions(channel, name=)`, `.tracks(channel)`, `.stim_raw()`, `.stim_seg()`, `.history(channel, kind, n, name=)`, `.last_pattern()`, `.t`, `.time`, and the runtime setting requests `.settings(channel)`, `.set_exposure()`, `.set_config()`, `.set_property()`, `.set_position()` (applied from the next timepoint, recorded in `events.parquet` and as frames-table columns).
- `AcquisitionPlan` — the schedule as data; `events_at(t)`, `pattern_due()`, `pattern_requirements()`.
- `ImagingConfig` — holds MicroManager config groups and device properties for a channel; supports inheritance and override.
- `ExperimentSchedule` — aggregates all `Experiment` objects, positions, and timing.