# More than one SLM (DMD) per run: design note

Status: built by Harrison on the `multi-slm` branch (PR #53, opened
2026-11-18) and tested on the lab's two-DMD rig. On 2026-11-21 Harrison
asked whether it should live in main or on a separate branch kept alongside
it. The recommendation was main, because the feature is opt-in by
configuration and a long-lived branch would have to be rebased against the
files that change most. Harrison agreed the same day. The branch was
rebased onto main after #54 (pattern chosen at use time). This note records
the design after the fact, following the convention of
docs/distribution-and-release-process.md §4. The decisions in §3 are the
ones the branch implements.

## 1. What it is for

A microscope can carry several DMDs, each fed by its own light source. Before
this change PyCLM sent every pattern to MicroManager's default SLM
(`Core-SLM`) through one camera-to-DMD affine. Two experiments in one run can
now stimulate through two DMDs, for example two wavelengths on two light
paths, each with its own closed loop.

## 2. What a user writes

`pyclm_config.toml`. The top-level `affine_transform` and
`slm_shape_h` / `slm_shape_w` stay the default SLM's. Each further DMD is a
table keyed by its MicroManager device label:

```toml
parallel_device_initialization = false   # optional, §3.5

[slm.DMD_L]
affine_transform = [[0.55, 0.006, 188.8], [-0.015, 1.11, 81.2]]
slm_shape_h = 684
slm_shape_w = 608
```

The experiment TOML:

```toml
[stimulation]
slm = "DMD_L"          # omit for MicroManager's default SLM
exposure = 100
    [stimulation.config_groups]
    Channel = "DMD_L_405"
```

User documentation: `documentation/first_time_setup.md` ("More than one
DMD") and `documentation/experiment_tomls.md` (`slm`).

## 3. Decisions

1. **One experiment drives one DMD.** The plan has one stimulation channel
   per experiment, and the DMD belongs to that stimulation
   (`Stimulation.slm`, `src/pyclm/schema.py`). Stimulating one experiment
   through two DMDs at once would need two `[stimulation]` tables and is a
   larger change to the plan. It is left out.
2. **The default SLM keeps the top-level keys.** A configuration without
   `[slm.<label>]` tables behaves exactly as before, so existing
   `pyclm_config.toml` files and data are unchanged. `PyclmConfig.slm` is a
   dict of `SlmConfig` (`schema.py:613`), and `slm_calibrations()`
   (`schema.py:739`) turns it into `{label: (shape, affine)}`.
3. **The calibration goes per experiment, not per device.**
   `Controller.initialize(slms=...)` (`controller.py:163`, from line 182)
   resolves each experiment's label to `(shape, affine)`. `SLMBuffer` keeps
   them as `calibrations` and `_calibration(name)` (`core/manager.py:185`)
   falls back to the default. `pattern_to_slm` takes the experiment name
   (`manager.py:190`). The microscope keeps `slm_devices`
   (`{experiment: label}`) and `_slm_for` (`core/microscope.py:241`) picks
   the device for `setSLMImage`, for grid tiles too. Each writer records an
   experiment with its own shape and affine, so `pyclm export` and
   `pyclm.io` need no change.
4. **Refuse a table that names the default SLM while other experiments use
   the default** (`declare_slm`, `microscope.py:187`). MicroManager makes
   the SLM loaded last its default. On the test rig that silently sent
   every pattern to one DMD. When every experiment names its own SLM there
   is nothing to confuse, and the table is accepted.
5. **`parallel_device_initialization`** (default true). False switches off
   MMCore's `ParallelDeviceInitialization` feature in `RealMicroscopeCore`,
   so devices initialize in `.cfg` order. On the test rig, parallel
   initialization of the two Polygons crashed the process.
6. **No rig-specific code in the repository.** The patched adapters the rig
   needed are recorded as docs/known-issues.md 50.

## 4. Interaction with #54 (pattern chosen at use time)

#54 (merged 2026-11-21) changed when the pattern is chosen. It did not
change where the pattern goes, and the multi-SLM commit rebased onto it
without conflicts. In detail:

- The SLM buffer now keeps a bounded history of DMD-space patterns per
  experiment and answers the microscope's request at use time. Every
  pattern that enters that history goes through
  `pattern_to_slm(..., experiment_name)` in `_handle_from_pattern`
  (`manager.py:212`), so each experiment's history is already in its own
  DMD's shape and affine. The blank placeholders are made with
  `_calibration(name)`.
- The microscope sends its request on `microscope_to_slm_buffer`, or on
  `microscope_to_pattern` for a `render_at_stimulation` method. It then
  applies the reply with `_slm_for(up_event.experiment_name)`. The event's
  experiment decides the device, and that is the same before and after #54.
- `first_pattern_wait_s` (now 15 s) and the held handshakes are per
  experiment, so an experiment on a second DMD waits for its own first
  pattern.
- #54's writer change affects only the tables (`tracks.parquet`,
  `frames.parquet`). The per-experiment affine and shape are written when
  the stores are opened, which #54 did not touch.

After the rebase the full suite passed on the branch (576 passed, 21
skipped), including `tests/test_multi_slm.py` and
`tests/test_pattern_at_use_time.py`. No run on the rig has combined the two
changes yet (§5).

## 5. Testing guide

**Dry:**

- `uv run --group test pytest tests/test_multi_slm.py
  tests/test_pattern_at_use_time.py`.
- Add an `[slm.second]` table to the demo's `pyclm_config.toml` and set
  `slm = "second"` in one experiment. Then run
  `uv run pyclm check <dir>` (it lists each table, its camera footprint and
  the experiments using it) and `uv run pyclm run <dir> --dry`. Read the
  output back with `pyclm.io` and confirm that the experiment's
  `affine_transform` is the table's.

**At the microscope:**

1. `pyclm check <dir> --mm-config <the .cfg>`. Each `[slm.<label>]` should
   be found in the `.cfg`.
2. First light on each DMD separately, one experiment at a time, with a
   uniform sample, to confirm that the pattern lands where it should and on
   the intended DMD (known-issues 50).
3. A run with two experiments on two DMDs. PR #53's runs (`yinyang`,
   `dual_demo`) correlated 0.80 to 0.96 with the projected patterns.
4. Repeat 3 on the merged main with #54. Confirm `pattern_t` in the frames
   table, and that both DMDs are dark after the run.
