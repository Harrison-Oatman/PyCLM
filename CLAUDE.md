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

# Run pyclm
uv run pyclm <experiment_directory> [--config path/to/pyclm_config.toml] [--dry]

# Convert HDF5 outputs to TIFF
uv run convert_hdf5s
```

**Linting:** ruff (formatter + linter with rules B, I, RUF, PT, UP) and nbstripout for notebooks — configured in `.pre-commit-config.yaml`.

**Optional dependency groups:**
- `cellpose` — CellposeSAM segmentation (requires torch/CUDA)
- `calibration` — DMD calibration via pycromanager
- `analysis` — data analysis (matplotlib, pandas, laptrack, seaborn, etc.)
- `dev` — analysis + docs + pytest

## Architecture

PyCLM is a closed-loop microscopy system that runs multiple simultaneous optogenetic experiments. The core architecture is a multi-threaded pipeline of processes communicating via queues.

### Process Pipeline

`Controller` (`controller.py`) owns all processes and runs them in a `ThreadPoolExecutor`. The processes are:

1. **`Manager`** (`core/manager.py`) — Timing brain. Iterates over timepoints and experiments, schedules acquisition/stimulation events, sends pattern generation requests. First to finish in a normal run; its exit triggers graceful shutdown.
2. **`MicroscopeProcess`** (`core/microscope.py`) — Controls hardware via pymmcore-plus. Executes acquisition events, updates stage positions, applies SLM patterns.
3. **`MicroscopeOutbox`** (`core/manager.py`) — Writes data to per-experiment HDF5 files (SWMR mode). Routes acquired data to segmentation and pattern processes as needed.
4. **`SLMBuffer`** (`core/manager.py`) — Holds the current DMD pattern per experiment, applies the affine transform from camera to SLM coordinates, sends patterns to the microscope on demand.
5. **`SegmentationProcess`** (`core/segmentation_process.py`) — Runs cell segmentation on acquired images.
6. **`PatternProcess`** (`core/pattern_process.py`) — Generates light patterns using registered `PatternMethod` instances.

All inter-process communication uses queues defined in `core/queues.py` (`AllQueues`). All processes extend `BaseProcess` (`core/base_process.py`), which polls registered queues and sleeps when idle to avoid busy-waiting.

### Experiment Configuration

Experiments are configured entirely via TOML files in an experiment directory:

- **`[experiment].toml`** — One per experiment. Defines imaging channels, stimulation, segmentation method, and pattern method with kwargs.
- **`schedule.toml`** — Timing: `[timing]` section with `steps`, `interval_seconds`, `setup_time_seconds`, `time_between_positions`.
- **`multipoints.xml`** — Imaging positions exported from MicroManager's multipoint list. Position labels link to experiment TOMLs (e.g., position `feedbackexp.1` uses `feedbackexp.toml`).
- **`pyclm_config.toml`** — Hardware config: `config_path` (MicroManager .cfg), `affine_transform` (2×3 matrix, camera→SLM), `slm_shape_h`/`slm_shape_w`. Located at repo root or in the experiment directory.

`directories.py:schedule_from_directory()` parses all of the above into an `ExperimentSchedule`.

### Extending PyCLM

**Custom pattern methods:** Subclass `PatternMethod` (`core/patterns/pattern.py`). Implement `generate(self, context: PatternContext) -> np.ndarray` returning a float array (0–1) in camera coordinates. Use `add_requirement(channel_name, raw=True/False, seg=True/False)` in `__init__` to declare what data the method needs. Register with `Controller.register_pattern_method(name, cls)` before `c.initialize()`.

**Custom segmentation methods:** Subclass `SegmentationMethod` (`core/segmentation/segmentation.py`). Register with `Controller.register_segmentation_method(name, cls)`.

**Virtual microscope (dry run):** `--dry` flag activates `SimulatedMicroscopeCore` + `TimeSeriesImageSource` (feeds from a `tif-source/` folder). Useful for testing pattern logic without hardware.

### Data Output

Each experiment produces one `.hdf5` file. Datasets are organized by timepoint and channel (e.g., `00001/channel_GFP/data`). Stimulation patterns are saved under `stim_aq/dmd`. SWMR mode allows reading files while the experiment is running.

### Key Types

- `AcquisitionEvent` — encodes a single image acquisition: position, channel, exposure, flags (segment, save, needs_slm).
- `PatternContext` — passed to `PatternMethod.generate()`, provides `.raw(channel)`, `.segmentation(channel)`, `.stim_raw()`, `.stim_seg()`.
- `ImagingConfig` — holds MicroManager config groups and device properties for a channel; supports inheritance and override.
- `ExperimentSchedule` — aggregates all `Experiment` objects, positions, and timing.