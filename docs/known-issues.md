# Known issues

Concrete bugs, hazards, and smells found while tracing data flow through the
core (assessment of `7af035a`, September 2026). Numbers are stable because
[assessment-2026-09.md](assessment-2026-09.md) cites them; an entry that has
been fixed is collapsed to a one-line **Fixed** note rather than deleted.
Line numbers in open entries refer to the post-Stage-0 code on
`Stage0-hygiene`.

Severity: **bug** = wrong behaviour observed or provable; **hazard** = can
crash, hang, or lose data under plausible conditions; **smell** = works, but
will resist the planned changes.

---

## Confirmed bugs

### 1. `t_delay > 0` broke pattern generation

**Fixed in Stage 0** (2026-09-06). `RequestPattern.t_index` is now the absolute
timepoint, matching `AcquisitionEvent.t_index`; the cadence decision still
uses the experiment-relative index. Covered by
`tests/test_manager_scheduling.py::test_pattern_request_index_matches_acquisition_index`
and `tests/test_pattern_process.py`. The original symptom: with `t_delay = 1`
the last frame raised `KeyError` and earlier frames were paired with the next
timepoint's request, so patterns were generated from the wrong data.

### 2. Manager wait loop spun at 100 %

**Fixed in Stage 0.** `Manager.process` now drains its inbox and sleeps
`sleep_interval` (10 ms) between checks. Regression test:
`test_manager_scheduling.py::test_wait_loop_sleeps_instead_of_spinning`.

### 3. Logging was set up once per interpreter

**Fixed in Stage 0.** `set_logging` removes the handlers it installed on a
previous call, so each `run_pyclm()` logs to its own directory
(`tests/test_logging_setup.py`).

### 4. Forced shutdown never closed HDF5 files

**Fixed in Stage 0.** `MicroscopeOutbox.process` closes files in a `finally`,
and `Controller.run` calls `close_files()` in its own `finally`
(`tests/test_shutdown.py::test_forced_stop_closes_files`).

### 5. `PatternReview` could not be constructed through the normal path (method removed in Stage 6)

**Fixed in Stage 0.** Constructor now takes only `h5fp`, `channel`, `**kwargs`
(`tests/test_pattern_method.py`).

### 6. A frame whose dataset was not pre-allocated is dropped (bug, partly addressed)

The HDF5 writer counts the loss in `dropped_frames` and logs an error; the
OME-Zarr writer (Stage 2) has no pre-allocation problem for planned frames,
since arrays are sized from the plan and chunks are written on demand.
**Surfaced in Stage 4 (2026-09-08):** the writer's `dropped_frames`, every
process's `error_count` and the router's `undeliverable` count are in
`status.json` under `health`, rewritten every timepoint.

---

## Hazards

### 7. Microscope thread had no exception guard; SLM handshake could kill a run

**Fixed in Stage 0.** `MicroscopeProcess.process` catches per-message
exceptions, counts them (`error_count`, `consecutive_errors`) and continues;
after `max_consecutive_errors` (default 10) in a row it re-raises so the
Controller aborts. The SLM handshake discards stale replies and, on timeout,
keeps the current pattern with a warning (`tests/test_microscope_process.py`).

### 8. `PFSPositionMover` could hang forever

**Fixed in Stage 0.** The focus-lock poll sleeps 10 ms per iteration and
raises `TimeoutError` after `PFS_TIMEOUT_S` (30 s). With the microscope error
guard, that means one logged error and an out-of-focus acquisition for that
position rather than a stalled run. Not unit-tested against real PFS timing.

### 9. Fixed 1 s settle before every snap

**Fixed in Stage 0.** `settle_time_seconds` in `pyclm_config.toml` (default
1.0) → `Controller(settle_time_s=...)` → `MicroscopeProcess.settle_time_s`.
Still a single global value; per-channel settle is a Stage 1 plan concern.

### 10. Hardware names hard-coded in generic code (hazard, partly addressed)

The focus device is now `focus_device` in `pyclm_config.toml` (default
`"ZDrive"` for backwards compatibility, logged at INFO when the key is
absent). Still hard-coded: PFS device/property names as overridable class
attributes on `PFSPositionMover`; the dummy SLM shape `1140×900` in
`MicroscopeProcess.declare_slm` (config says 912). The y-axis negation
`PFSPositionMover` used to apply was removed on 2026-09-14: it compensated
for a wrong MicroManager stage configuration, not a Nikon convention.
Frames tables written before that date record the position list's `y`
while the stage actually went to `-y`.

### 11. Swallowed handler exceptions have no health signal (hazard, partly addressed)

Every `BaseProcess` and the microscope now count errors (`error_count`), and
the writer counts `dropped_frames` and the router `undeliverable`, but nothing
reads those counters during a run. A pattern method that throws every timepoint still means the SLM keeps
showing the last good pattern with no indication in the GUI or to the
Manager. **Surfaced in Stage 4:** `status.json` carries the counters and the
GUI shows a status line; acquisition failures are also `acquisition_error`
rows in `events.parquet`.

### 12. `multiprocessing.Queue` between threads

**Fixed in Stage 0.** `AllQueues` uses `queue.Queue`; items are passed by
reference, `empty()`/`get_nowait()` are reliable, and the unused
`outbox/slm/seg/pattern → manager` queues were removed.

### 13. Pre-existing output files were detected last

**Fixed in Stage 0.** `Controller.initialize` checks for existing
`<experiment>.hdf5` files before any method or model is built
(`tests/test_controller_init.py`). No `--overwrite` flag yet.

---

## Smells that will resist the planned features

### 14. "Is channel X scheduled at t?" was implemented four times

**Mostly fixed in Stage 1.** `Manager.process`, `MicroscopeOutbox.initialize`
and `_timepoint_complete` now all derive from `AcquisitionPlan`
(`is_scheduled`, `events_at`, `datasets_at`). The GUI's
`ChannelSchedule.is_scheduled_at` (`gui/gui_controller.py`) is the remaining
copy; it reads the same `every_t`/`t_delay`/`t_stop` attributes the plan
writes, and Stage 2 hands it the plan via the file.

### 15. The HDF5 attribute schema is written twice

`AcquisitionEvent.as_attrs` (`core/events.py`; Stage 0 merged `write_attrs`
and `__repr__` onto it) and `MicroscopeOutbox._preallocate_attrs`
(`core/manager.py`), which must list the same keys so SWMR readers see them
(Stage 1 replaced `sub_axes` with `index` in both). Stage 2 (storage v2)
replaces per-dataset attributes with a frames table.

### 16. Helper functions copied between modules

**Fixed in Stage 2.** `pyclm.io` reads both layouts; the GUI and
`convert_hdf5s` use it, and the duplicated helpers are gone.

### 17. Shutdown counts encode the topology

**Fixed in Stage 3.** `Router.end_stream` derives each consumer's fan-in from
the resolved table (`core/router.py`); no process counts stream closes
(`tests/test_router.py::test_stream_close_arrives_after_the_last_upstream`,
`tests/test_shutdown.py`).

### 18. Routing is baked into `AcquisitionEvent`

**Fixed in Stage 3.** Events carry identity only; the `Router` resolves who
receives what from the processes' declarations (`core/router.py`,
architecture-notes §1). Adding a consumer is `Controller.add_process()`;
`TrackingProcess` was added that way.

### 19. `pattern_shape` became a float tuple after binning

**Fixed in Stage 0.** `PatternMethod.update_binning` reconstructs the unbinned
shape and rebins with integer division (`tests/test_pattern_method.py`).

### 20. Dead or vestigial code

**Fixed in Stage 0.** Removed `GeneratePatternEvent`, the
`"initialize_slm_queue"` message case, the `PositionGrid` stub, the four
unused `*_to_manager` queues, `DataPassingProcess.message_history`, stale
`todo` comments, and unused imports (ruff `F401` clean).
`PositionWithAutoFocus` stays because `positions_from_xml` uses it.

### 21. Method registries were class-level

**Fixed in Stage 0.** `PatternProcess` and `SegmentationProcess` copy their
registry per instance (`tests/test_pattern_process.py::test_registration_is_per_instance`).

### 22. Empty `seg` datasets are always allocated

`SegmentationConfig("none")` inherits `save_output=True`
(`directories.py`, `core/experiments.py:MethodBasedConfig`), so
`MicroscopeOutbox.initialize` pre-allocates `.../seg` datasets for every
channel of every experiment even with no segmentation method.
`tests/test_dry_run.py` codifies the empty datasets as expected output.
Change with the Stage 2 layout, not before (it alters the file format).
Stage 1 fixed a consequence: `convert_hdf5s.make_tif` crashed with "all
input arrays must have the same shape" when it stacked one of these empty
`seg` datasets next to a real frame; it now treats a `seg` dataset whose
shape differs from the frame as absent (`tests/test_swmr.py` covers it).

### 23. Segmentation only exists if a pattern asks for it

**Addressed in Stage 0.** Behaviour unchanged (it is a reasonable
optimisation), but `Controller.initialize` now logs a warning when a
segmentation method is configured and no pattern requirement uses it
(`tests/test_controller_init.py::test_unused_segmentation_method_warns`).
Stage 3 made this the general rule of the Router: a producer stage
(segmentation, tracking) runs for a channel only when a demanding consumer
needs its output; `save` only decides whether the writer records it
(`Router.resolve`, docs/stage3-router-design.md §3.4, decision 2).

### 24. Virtual microscope fidelity around binning

`SimulatedMicroscopeCore.getROI` returns 4× the TIF size
(`core/virtual_microscope/simulated_core.py`) and `snapImage` never bins
(commented out), so dry runs with `binning != 4` produce frames of a different
size than a real camera would, and `maxshape` is silently larger than the
data. The GUI's fallback layer is hard-coded `(1, 800, 800)`
(`gui/gui_controller.py`).

### 25. `print()` in per-timepoint hot paths

**Fixed in Stage 0.** Core modules log through module loggers. Deliberately
kept as `print`: the operator progress line `t = N: M minutes` and `DONE` in
`Manager.process`, the startup listing in `run_pyclm`, and
`SimulatedMicroscopeCore.describe()`.

### 26. Manager inbox is only drained between timepoints

`Manager.drain_inboxes` runs inside the inter-timepoint wait loop
(`core/manager.py`). Fine for the z-correction message, but a control plane
(pause, set position, update parameters) needs the Manager to process
commands at defined points and acknowledge them. **Stage 4 (2026-09-08):**
that loop is now the defined point. Setting requests from pattern methods
and the microscope's acknowledgements are drained there and applied from
`current_t`; the Manager never mutates state mid-burst.

### 27. `experiment_from_toml` mutated the parsed TOML

**Fixed in Stage 0.** The `[segmentation]` and `[pattern]` tables are copied
before `method` is popped.

### 28. Lint debt

**Fixed in Stage 0.** `pre-commit run --all-files` (ruff format + check,
nbstripout) passes.

### 29. One segmentation configuration per experiment

**Fixed in Stage 3 (2026-09-08).** `experiment_from_toml` built a single
`SegmentationConfig`, `SegmentationProcess` kept one model per experiment,
and the routing kind `seg` was one per channel, so two models on one frame
(nuclei and whole cells of a biosensor channel) or different models per
channel could not be configured. Named `[segmentation.<name>]` tables,
kind `seg:<name>`, `Experiment.segmentations` and one label image per table
in OME-Zarr (see stage3-router-design.md §9) remove the limit.

### 30. Per-channel `config_groups` / `device_properties` overrides were never applied

**Fixed in Stage 5 (2026-09-08).** `experiment_from_toml` looked for a
`config_groups` key *inside* `[channels.<preset>.config_groups]` (and the
same for device properties), so the documented per-channel tables were
always empty. The schema (`ExperimentConfig.to_experiment`) applies them;
`tests/test_schema.py` covers it.

### 31. `t_delay` / `t_stop` after `[pattern]` were pattern arguments

**Fixed in Stage 5 (2026-09-08).** A TOML key written after a table header
belongs to that table, so the documented placement of `t_delay` and
`t_stop` at the end of the file made them `[pattern]` keys that the
method's `**kwargs` swallowed; the experiment kept `t_delay = 0`. The
schema refuses them in any method table with a message saying where they
go, and the documented examples put them at the top of the file.

### 32. `RealMicroscopeCore` lost most of its methods to a mis-placed helper (Stage 6)

**Fixed 2026-09-12.** The device-interface hint added on 2026-09-09 put two
module-level functions into `core/real_core.py` between
`loadSystemConfiguration` and the rest of the class, so every method after
them (`getCameraDevice`, `getAvailableConfigGroups`, the stage, SLM and
camera calls) silently became a dead nested function and the interface's
stubs, which return None, answered instead. Only the real microscope was
affected; the suite never instantiates the real core. Found by Harrison on
the scope as `TypeError: argument of type 'NoneType' is not iterable` in
`set_binning`. The helpers now live below the class and
`tests/test_microscope_process.py` asserts that the real core defines every
interface method itself.

### 33. OME-Zarr metadata writes failed under a reader on Windows

**Fixed 2026-09-16.** Zarr writes `.zgroup` / `.zattrs` / `.zarray` by
writing a `.partial` file and replacing the target; Windows refuses the
replace while another process has the target open, and the viewer
re-reads those files every second. Seen on the scope as `PermissionError:
[WinError 5] Access is denied … .zgroup.<id>.partial -> .zgroup` from
`_update_progress`; the frame itself was already written and the next
frame's progress update healed `current_t`, so no data was lost, but the
error was counted and the `.partial` file left behind. Every metadata
write (progress, pattern ids, array resizes) and the parquet table move
now retry with a short backoff (`_retry_write`, `ome_zarr.py`), as the
HDF5 writer already did for its own Windows collisions.

### 34. A PFS that switched itself off stayed off for the rest of the run

**Mitigated 2026-09-17, to be confirmed on the scope.** Overnight, the
Ti2's PFS went to "off" (the stage controller's display) part-way through
a run and every later position waited the full 30 s and raised
`TimeoutError` (2097 times); the status polled was `0000001100001001`
throughout, against `...1010` when locked, so the low bits are the
on/off state and not another spelling of "locked". The mover had set
`FocusMaintenance = "On"` at every position, which did nothing; pressing
"on" at the controller restored it at once, so the sample was in range.
The likely reason is that the adapter (or MMCore's cache) still held "On"
and did not resend it. `PFSPositionMover` now sets "Off" then "On" every
`PFS_RETRY_S` (5 s) while it waits, and logs a warning for each restart.
Not yet known: what switches the PFS off (the interface lost during a
large z excursion is the usual cause; see the "Major Z position change"
warnings in the same logs). If the restart does not bring it back, the
next thing to try is `core.enableContinuousFocus(True)` / `fullFocus()`.

### 35. Dry runs with `binning` in `dry_run.yml` store per-cell camera patterns at the wrong scale

**Open, found 2026-10-01 while building `pyclm new --demo`.** The virtual
microscope reports a camera `ROI_FACTOR` (4) times the TIF, so a method's
`pattern_shape` is 4× the frames when the experiment's binning is 1.
`dry_run.yml` `binning: b` compensates only in the SLM buffer (the affine
is scaled by `b`), so the DMD pattern comes out right; but a method that
paints from the frames' labels (`move_out`) draws into the top-left
quarter of its 4×-sized canvas, and that is what `patterns/camera` stores
(seen: 0.6 % of the camera pattern lit against 10 % of the DMD). Patterns
that fill the field (bars) hide it. Working setup, used by the demo:
experiments at `binning = 4`, `dry_run.yml` with the unbinned
`pixel_size_um` and no `binning`; frames, patterns and both stored
patterns then agree. Fix to decide: make the dry `binning` set the
simulated camera's reported binning (so the pattern shape follows), or
retire the key in favour of the TOML binning.

### 36. `fb_bounce` reversed only once

**Fixed 2026-10-26** (method zoo PR, `docs/method-zoo-design.md`).
`BounceModel.generate` (`src/pyclm/core/patterns/fbc_cell_movement.py`)
set `self.down = False` after the first half-loop and never back, so after
`t_loop / 2` minutes it lit the same side of every cell for the rest of
the run. It now follows the half of the loop it is in, like `bar_bounce`;
`tests/test_zoo_gallery.py` checks it against `direction` with
`sequence = "ud"`.

### 37. Per-cell `gradient = true` turned a one-pixel-wide cell into NaN

**Fixed 2026-10-26** (method zoo PR). `PerCellPatternMethod.prop_vector`
(`src/pyclm/core/patterns/fbc_cell_movement.py`) divided by the largest
projection in the cell; a cell with no pixel ahead of its centroid (a
one-pixel label) divided 0 by 0, and the NaN reached the pattern. Such a
cell now stays dark; `tests/test_zoo_gallery.py` checks it.

### 38. `embryo_inner` / `embryo_outer` crashed on a non-square field

**Fixed 2026-10-26** (keyword-conventions PR,
`docs/pattern-methods-expansion-design.md`). `PatternAlongAxis.generate`
(`src/pyclm/core/patterns/embryo_patterns.py`) built its grid with
`np.meshgrid(y, x)` (default `xy` indexing), which is `(w, h)`; multiplying
by the `(h, w)` mask raised `ValueError: operands could not be broadcast
together with shapes (100,60) (60,100)` on a 60 × 100 field. A square
field worked because the swapped axes and the swapped centroid indices
cancelled. It now uses `indexing="ij"` with the major axis as a
`(row, column)` vector; patterns on square fields are unchanged (checked
pixel for pixel against the old code), and
`tests/test_keyword_migration.py` covers a 60 × 100 field.

### 39. `nuc_channel` was silently ignored

**Fixed 2026-10-26** (same PR). Early nucleus-clamp TOMLs (the archived
`resources/tomls/tag-rfp/*.toml`) set `nuc_channel = "545"`, which the
methods swallowed through `**kwargs`; the run used the default `channel`,
which happened to be `"545"` too. `nuc_channel` is now an old name of
`channel`, so it takes effect, and `pyclm check --fix` renames it.

### 40. Unchanged config groups were set again before every frame

**Fixed 2026-10-28** (session 3 fixes PR). `MicroscopeProcess.handle_config_update`
(`src/pyclm/core/microscope.py`) called `setConfig` for every group of
every frame. On the lab's rig, setting a group already at its preset
cost 32 ms (System) to 167 ms (Channel) with `waitForSystem`, about 0.35 s
a frame (session 3 run 1, `run1_timing`). The microscope now remembers the
preset it last set per group and skips groups already there; device
property updates reset the memory. `reapply_config_groups = true` turns it
off. `tests/test_microscope_process.py` covers both.

### 41. The first stimulation got the blank placeholder

**Fixed 2026-10-28** (same PR). `SLMBuffer` (`src/pyclm/core/manager.py`)
answers each update-pattern handshake when the Manager dispatches it, with
whatever pattern it holds. Before an experiment's first pattern arrives
that is the blank placeholder (id 0), so a stimulation-only visit was
skipped as blank. The six dry-run HDF5 tests (`tests/test_dry_run.py`)
failed on main for this reason, since before #43: at t = 1 both bar
experiments got id 0 while their patterns (3200 × 3200 at the dry-run
scale) were still being made. The buffer now holds an experiment's
handshakes until its first pattern arrives (at most
`first_pattern_wait_s`, 5 s), and `_await_slm_pattern` keeps a reply for
a later event instead of discarding it, since held replies can come out of
order across experiments. A pattern made after dispatch is still used one
timepoint late when the run is behind (the "one segmentation behind"
Harrison accepted, 2026-10-28).

### 42. The DMD kept the previous image at the start of a run

**Fixed 2026-10-28** (same PR). Nothing set the DMD before the first
stimulation; in session 3's DMD test the first frame after an all-zero
image still showed light on a third of the field. `MicroscopeProcess`
now blanks the SLM when it is declared and on `close`. Whether the
Mightex adapter reports Busy while it loads an image (so `waitForSystem`
covers it) is still to be checked on the scope.

### 43. Binned patterns were shifted by (binning - 1) / 2 unbinned pixels

**Fixed 2026-10-28** (same PR). `compose_affine` scaled the linear part by
the binning, which puts binned pixel `p` at unbinned `binning * p`; its
center is at `binning * p + (binning - 1) / 2`. Session 3 measured the
shift at binning 2 (0.25 binned px, 0.17 um at 20x). The dry run's
`dry_run.yml` binning scaling (`Controller.initialize`) has the same form
and is left as it is with #35.

### 44. `pyclm check` named the wrong pixel-size preset

**Fixed 2026-10-28** (same PR). `MMConfig` (`src/pyclm/mmconfig.py`) did
not read the `.cfg`'s `Label,<device>,<state>,<label>` lines, so a
pixel-size preset conditioned on `Nosepiece-State` never matched an
Objective preset that sets `Nosepiece-Label`; ties then sorted by name,
and check printed the 10x preset for 20x experiments. Runs read
`getPixelSizeUm()` and were not affected.

### 45. `pyclm check` suggested a camera_roi larger than the sensor

**Fixed 2026-10-28** (same PR). The suggestion clipped the footprint's
origin to 0 but kept its full size ([0, 0, 1971, 3152] on a 2048 × 2048
camera). `suggest_camera_roi` (`src/pyclm/schema.py`) now shrinks the size
with the origin; check cannot see the sensor offline, so `pyclm run`
logs the ROI clipped to the camera and names the edges the DMD does not
reach.

### 46. A methods file that is also an entry point was "defined twice"

**Fixed 2026-10-28** (same PR). Importing one file by path and through an
installed entry point makes two class objects; `MethodSet.add`
(`src/pyclm/methods.py`) compared them by identity. Classes from the same
source file with the same qualified name are now one method, with a note.

### 47. A run died on Windows when its output went to a file

**Fixed 2026-10-28** (same PR). `run_pyclm` calls pymmcore-plus's
`core.describe()`, which prints an emoji; a redirected stdout on Windows is
cp1252 and raised `UnicodeEncodeError` before the first frame.
`pyclm.console.utf8_console()` reconfigures stdout and stderr to UTF-8
(`errors="replace"`) in `pyclm` and `run_pyclm`.
