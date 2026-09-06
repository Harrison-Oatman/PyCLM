# Known issues

Concrete bugs, hazards, and smells found while tracing data flow through the
core (branch `core-refactor`, HEAD `7af035a`, September 2026). Line numbers
are from that revision. Delete an entry when it is fixed.

Severity: **bug** = wrong behaviour observed or provable; **hazard** = can
crash, hang, or lose data under plausible conditions; **smell** = works, but
will resist the planned changes.

---

## Confirmed bugs

### 1. `t_delay > 0` breaks pattern generation (bug, verified by dry run)

`Manager` sends `RequestPattern` with the **experiment-relative** index
(`this_t = t - t_delay`, `core/manager.py:811,819-821`) but stamps
`AcquisitionEvent.t_index` with the **absolute** `t` (`manager.py:853,888`).
`PatternProcess` keys its `DataDock` by `f"{experiment}_{t_index:05d}"` on both
sides (`core/pattern_process.py:166` vs `:192,208`), so frames look for a dock
that was created under a different name.

Reproduced with a 4-step dry run, `t_delay = 1`, imaging `every_t = 1`, and a
pattern method requiring raw `545`: `generate()` ran 2× instead of 3×, the
last frame raised `KeyError: 'bar10.00_00003'` inside the pattern process, and
because the request for timepoint *t+1* is created before the frame for *t*
arrives, frames for *t* landed in the dock for *t+1*, i.e. patterns were
generated from the wrong timepoint's data without any error. Script:
`repro_tdelay.py` in the session scratchpad (recreate from the description).

Fix: use one index everywhere. Short term, send `t` in `RequestPattern`. Long
term, give each planned acquisition an identifier from the plan (see the
assessment, seam A) and key docks by it.

### 2. Manager wait loop spins at 100 % (bug)

`Manager.process` waits for the next timepoint with a `while` loop that
drains inboxes and checks `stop_event` but never sleeps
(`core/manager.py:792-800`). Between timepoints, which is most of an
experiment, one thread burns a core and contends for the GIL with the
segmentation and pattern threads. `BaseProcess` sleeps 1 ms when idle; the
Manager should do the same.

### 3. Logging is set up once per interpreter, not per run (bug)

`set_logging()` calls `logging.basicConfig(handlers=[...])`
(`run_pyclm.py:21-39`). `basicConfig` is a no-op when the root logger already
has handlers, so a second `run_pyclm()` in the same process keeps writing to
the first experiment directory's `log.log`. Affects tests, notebooks, and any
future in-process GUI that launches runs. Fix: attach/detach a file handler
explicitly per run.

### 4. Forced shutdown never closes HDF5 files (bug, low impact)

On `KeyboardInterrupt` or a crashed process, `Controller.run` sets
`stop_event` and waits (`controller.py:216-240`); `BaseProcess.process` just
breaks. `MicroscopeOutbox.close_files()` (`core/manager.py:293`) is only
called on the graceful path. Data survives because every write is followed by
`flush()`, but files are left to the garbage collector. Fix: `try/finally`
around the outbox loop or call `close_files()` in `Controller.run`'s `finally`.

### 5. `PatternReview` cannot be constructed through the normal path (bug)

`PatternProcess.request_method` instantiates methods as
`model_class(**experiment.pattern.kwargs)` (`core/pattern_process.py:71`), but
`PatternReview.__init__` requires positional `experiment_name` and
`camera_properties` (`core/patterns/pattern.py:271-278`). Registered as
`"pattern_review"` in `known_models`, so a TOML that selects it fails at
startup. Fix the signature or remove the class.

### 6. A frame whose dataset was not pre-allocated is dropped silently (bug)

`write_data` does `f[relpath + dset_name]` (`core/manager.py:439`); a missing
key raises `KeyError`, caught by the blanket handler at `:482` and logged. The
frame is gone and nothing upstream is told. This is the failure mode for any
schedule/HDF5 mismatch (issue #1's cousin) and for future runtime schedule
edits. Fix: create-on-demand where safe, and surface a counter/warning to the
Manager and GUI.

---

## Hazards

### 7. Microscope thread has no exception guard; the SLM handshake can kill a run

`MicroscopeProcess.process` (`core/microscope.py:74-119`) dispatches without a
`try/except`, unlike `BaseProcess`. `handle_update_pattern_event` blocks on
`slm_queue.get(True, 5)` and `assert`s the event id matches
(`microscope.py:200-205`). A slow SLM buffer (>5 s) raises `queue.Empty`; a
stale `EventSLMPattern` left in the queue by an earlier timeout raises
`AssertionError`. Either kills the microscope future, and `Controller.run`
aborts the whole experiment. A hardware error from pymmcore in any handler
does the same. Decide per error class whether to skip the event, retry, or
abort, and log rather than assert.

### 8. `PFSPositionMover` can hang forever

The focus-lock wait is `while status != LOCKED: pass`
(`core/position_mover.py:93-97`): no sleep, no timeout, no stop-event check.
If PFS never locks (sample edge, air bubble), the run stalls with no message.

### 9. Fixed 1 s settle before every snap

`microscope.py:243` sleeps 1.0 s unconditionally after `waitForSystem()`.
With 2 channels + stim per position that is already 3 s of the per-position
budget; z-stacks and grids multiply it. Make it configurable per channel or
derive it from the plan.

### 10. Hardware names hard-coded in generic code

`core.setFocusDevice("ZDrive")` in `run_pyclm.py:143`; PFS device/property
names as class attributes in `PFSPositionMover` (documented as overridable);
y-axis negation in `position_mover.py:82`; dummy SLM `1140×900` in
`microscope.py:61-62` (config says 912). Focus device belongs in
`pyclm_config.toml`.

### 11. Swallowed handler exceptions have no health signal

`BaseProcess.process` logs and continues (`core/base_process.py:71-74`). A
pattern method that throws every timepoint means the SLM keeps showing the
last good pattern with no indication in the GUI or to the Manager. The
`pattern_id` provenance makes this recoverable after the fact, but an
experiment can run for hours in that state. Add a per-process error counter
and a status message to the Manager (and eventually the GUI).

### 12. `multiprocessing.Queue` between threads

`core/queues.py` uses `multiprocessing.Queue` while `Controller.run` uses a
`ThreadPoolExecutor`. Every frame is pickled through a pipe (copy, ~7 ms per
8 MB frame, measured), `empty()` is racy (a `put()` is invisible until the
feeder thread runs), and `Manager.process` relies on `empty()` followed by a
blocking `get()` (`manager.py:794-795`). Not a bottleneck today; it is a
hedge between two concurrency models that pays the costs of both. Switch to
`queue.Queue` unless a real move to processes is planned.

### 13. Pre-existing output files are detected last

`MicroscopeOutbox.initialize` raises `FileExistsError` (`core/manager.py:169-172`)
after all methods and models (including Cellpose on GPU) have been built, and
a run that crashes after initialisation leaves files that block the next run
until deleted by hand. Check before heavy initialisation; consider an
`--overwrite` flag or timestamped output directories.

---

## Smells that will resist the planned features

### 14. "Is channel X scheduled at t?" is implemented four times

`Manager.process` (`core/manager.py:803-816,828,866`), `MicroscopeOutbox.initialize`
(`:189-198`), `MicroscopeOutbox._timepoint_complete` (`:382-420`), and the GUI's
`ChannelSchedule.is_scheduled_at` (`gui/gui_controller.py:32-44`). They agree
today; any axis added (z, grid) or any runtime edit has to be made in all four.

### 15. The HDF5 attribute schema is written three times

`AcquisitionEvent.write_attrs` (`core/events.py:160-205`),
`AcquisitionEvent.__repr__` (`:207-254`, a copy), and
`MicroscopeOutbox._preallocate_attrs` (`core/manager.py:353-380`).

### 16. Helper functions copied between modules

`get_binning_from_metadata` and `find_affine_transform` exist in both
`gui/gui_controller.py:447-485` and `convert_hdf5s.py:72-89,222-241`;
`set_binning` in both `controller.py:87-103` and `core/microscope.py:153-175`.
A small `pyclm.io` module for reading PyCLM HDF5 files would absorb the first
two and give analysis notebooks a supported entry point.

### 17. Shutdown counts encode the topology

`MicroscopeOutbox.handle_message` waits for `stream_count >= 2`
(`core/manager.py:331-349`), `PatternProcess` for `>= 2`
(`core/pattern_process.py:147-155`). Adding a consumer (tracking) means
auditing every count.

### 18. Routing is baked into `AcquisitionEvent` by the Manager

`Manager.get_kwargs` (`core/manager.py:668-714`) decides `segment`,
`save_seg`, `raw_goes_to_pattern`, `seg_goes_to_pattern` from the pattern's
`AcquiredImageRequest`s; the Outbox and Segmentation processes read the flags.
A new consumer needs a new flag, a new `get_kwargs` branch, a new branch in
the Outbox, and a new attribute in the HDF5 schema (#15).

### 19. `pattern_shape` becomes a float tuple after binning

`PatternMethod.update_binning` divides with `//` by a float ratio
(`core/patterns/pattern.py:247-261`), so every built-in method casts with
`int(h), int(w)`. Keep shapes integral.

### 20. Dead or vestigial code

`GeneratePatternEvent` (`core/events.py:46`), the `"initialize_slm_queue"`
message case (`core/manager.py:591-597`), the `PositionGrid` stub
(`core/events.py:257-265`), the four unused `*_to_manager` queues,
`DataPassingProcess.message_history` (unbounded list, `manager.py:68,107`),
`PositionWithAutoFocus` (kept for XML import), the `todo` comments in
`core/experiments.py:340-341`, unused imports (`Thread` in `run_pyclm.py`,
`active_count`/`sleep`/`as_completed` in `controller.py`).

### 21. Method registries are class-level

`PatternProcess.known_models` and `SegmentationProcess.known_models` are
`ClassVar` dicts mutated by `register_method` (`core/pattern_process.py:25,98`;
`core/segmentation_process.py:18,63`), so registrations leak between
`Controller` instances (tests, notebooks).

### 22. Empty `seg` datasets are always allocated

`SegmentationConfig("none")` inherits `save_output=True`
(`directories.py:127`, `core/experiments.py:108`), so
`MicroscopeOutbox.initialize` pre-allocates `.../seg` datasets for every
channel of every experiment even with no segmentation method
(`core/manager.py:212,248`). `tests/test_dry_run.py` codifies the empty
datasets as expected output.

### 23. Segmentation only exists if a pattern asks for it

`Controller.initialize` calls `SegmentationProcess.request_method` only when a
pattern requirement has `needs_seg` (`controller.py:137-139`). A TOML with
`[segmentation] save = true` and an open-loop pattern silently segments
nothing. Reasonable as an optimisation, surprising as behaviour; at minimum
warn.

### 24. Virtual microscope fidelity around binning

`SimulatedMicroscopeCore.getROI` returns 4× the TIF size
(`core/virtual_microscope/simulated_core.py:192`) and `snapImage` never bins
(`:165-174`, commented out), so dry runs with `binning != 4` produce frames of
a different size than a real camera would, and `maxshape` is silently larger
than the data. The GUI's fallback layer is hard-coded `(1, 800, 800)`
(`gui/gui_controller.py:173`).

### 25. `print()` in per-timepoint hot paths

`core/manager.py:682,695,761,764,788`, `core/patterns/pattern.py:84`
(`DataDock.get_awaiting` prints on every completeness check),
`core/pattern_process.py:168,211`, `core/segmentation_process.py:135`,
`core/manager.py:93,328`. They bypass the log file and drown real warnings on
the console. Use `logger.debug`.

### 26. Manager inbox is only drained between timepoints

`core/manager.py:792-796`. Fine for the z-correction message, but a control
plane (pause, set position, update parameters) needs the Manager to process
commands at defined points, and to acknowledge them.

### 27. `experiment_from_toml` mutates the parsed TOML

`segmentation.pop("method")`, `pattern.pop("method")`
(`directories.py:117,131`). Harmless, but it prevents re-serialising the
parsed config verbatim and is a symptom of having no config model.

### 28. Lint debt on this branch

`run_pyclm.py` imports are out of order (`import json` after third-party
imports; `logger` defined mid-import block), which ruff's `I` rule will flag
when `pre-commit` runs.
