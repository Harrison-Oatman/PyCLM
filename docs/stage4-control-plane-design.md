# Stage 4: runtime setting changes and feedback from the microscope — design

Date: 2026-09-08. Status: **implemented** the same day on `core-refactor`,
with every decision in §5 taken as recommended. Code:
`core/settings.py` (typed changes), `PatternContext.set_*` / `settings()`
/ `position()` (`core/patterns/pattern.py`), `SettingsRequestMessage` and
`EventDoneMessage` (`core/messages.py`), `Manager.apply_settings` /
`handle_event_done` / `status` / `finish` (`core/manager.py`),
`core/storage/events.py` (`EventLog`), override columns in the OME-Zarr
frames table, `ExperimentData.events` in `pyclm.io`, `status.json` and its
line in the GUI. Tests: `tests/test_settings.py`, a settings dry run, the
far-red example in `tests/test_doc_examples.py`.

Stage 4 is seam **E** of [assessment-2026-09.md](assessment-2026-09.md):
no control plane, no feedback from the microscope. Its scope was narrowed
twice during planning (§1), and the record below describes what was built.

---

## 1. Scope and how it was narrowed

The roadmap's Stage 4 was a general control plane: pause/resume, live
position and parameter edits, extending a run, hot-reload of the
experiment directory, acknowledgements, an events table. Two decisions
narrowed it:

1. **Structure and schedule stay fixed.** No pause, no extra timepoints, no
   cadence or channel changes. The plan built at start-up is the plan that
   runs; the OME-Zarr cadence groups it defines are never resized or
   re-mapped. Only *values* inside the plan change: an experiment's
   position and its channels' exposure, config presets and device
   properties. This is the intermediate between a static schedule and
   editing the schedule that the user asked for.
2. **The pattern method is the source of changes, not a file.** Editing
   TOMLs live was rejected because several positions share one TOML and
   because the thing that knows what an experiment needs next is its
   pattern method. So a method changes settings of *its own experiment*
   from inside `generate`, with the same one-interval delay as the pattern
   itself. The driving case is a red / far-red optogenetic tool: red light
   through the DMD, far-red delivered as a channel, both lasers switched
   through a device property by a programme the method follows.

Kept from the roadmap: acknowledgements from the microscope, a lateness
and error record, a status file, and an events table. Deferred: pause,
hot-reload, a command file or socket, a GUI with controls, skipping late
timepoints, multiple DMDs (the design keys every request by channel name
so a second stimulation channel later is additive).

---

## 2. Design

### 2.1 Requests (`core/settings.py`, `PatternContext`)

A `SettingChange` is `(kind, channel, key, value)`: `exposure`
(`exposure_ms`), `config` (key = group, value = preset), `property` (key =
`<device>-<property>`), `position` (key in `x`, `y`, `z`, `pfs_offset`;
no channel). The context offers `set_exposure`, `set_config`,
`set_property`, `set_position`, and read-back through `settings(channel)`
and `position()`. `"stimulation"` names the experiment's stimulation
channel. Requests accumulate on the context during one `generate` call.

### 2.2 Transport and timing

After `generate` returns, `PatternProcess.run_model` sends the collected
changes to the Manager as one `SettingsRequestMessage(experiment,
t_requested, changes)` on the new `pattern_to_manager` queue. The Manager
drains its inboxes only inside the inter-timepoint wait, so a request is
applied before the burst of the timepoint being awaited and never in the
middle of one; `Manager.next_timepoint()` is the first timepoint whose
events have not been emitted (it is `current_t + 1` once the loop has
ended, so a change that arrives too late is recorded as applying to a
timepoint that never runs, not to one already acquired). A request made while generating for `t`
therefore applies from `t + 1` when segmentation is fast, or later when it
is slow; both timepoints are recorded.

### 2.3 Application (`Manager.apply_settings`)

Structural validation (`settings.check_change`: known kind, positive
finite exposure, non-empty group/preset/device/property, finite position)
plus "channel exists" against the plan. A refused change logs a warning
and an event row; the others in the same request still apply. An applied
change mutates the experiment's live `ImagingConfig` (the same object the
plan returns from `imaging_config`) or `MicroscopePosition`, so the next
burst carries it. Hardware rejection (an unknown preset at
`core.setConfig`) surfaces through the acknowledgement path as an
`acquisition_error` event; the Manager cannot validate against the
MicroManager configuration because the core belongs to the microscope
thread.

### 2.4 Provenance

- **Frames table.** The Manager keeps `overrides[(experiment, channel)] =
  {column: value}` for config and property changes and stamps a copy on
  every later `AcquisitionEvent` (`event.overrides`). The OME-Zarr writer
  adds one frames-table column per key ever seen (`<device>-<property>` or
  the config group name), null on frames before the first change.
  Exposure and position already had columns.
- **Events table.** `events.parquet` in the experiment directory (rewritten
  per row, `events.csv` at close): wall time, `t_requested`, `t_applied`,
  experiment, channel, kind, key, old, new, status (`applied`, `refused`,
  `failed`, `warning`), detail. Also holds `z_correction` rows from the
  focus lock, `late` rows and `acquisition_error` rows. `pyclm.io`
  exposes it as `ExperimentData.events`, filtered by experiment.
- The original `plan.useq.yaml` is untouched; the events table is the
  history.

### 2.5 Acknowledgements and status

The microscope sends `EventDoneMessage(event, error=None)` after every
acquisition, and one with the error text when an acquisition fails (from
its error guard). The Manager records per timepoint and experiment the
frames done, errors and the worst lateness (`completed − scheduled`), warns
once per timepoint when lateness exceeds the interval, and writes
`status.json` before each burst and once more at the end
(`Manager.finish`, called by the Controller after every process has
exited, so the last acknowledgements are absorbed): progress, per
experiment last timepoint / lateness / errors, counts of applied and
refused settings, and process health (`error_count` per process,
undeliverable frames, dropped frames) from a Controller callback. The GUI
shows one line from it in napari's status bar.

---

## 3. What did not change

The plan and its YAML, cadence groups and storage layout, the router, the
TOML format, `generate(context)`. HDF5 format 1 records the changed values
in its per-dataset attributes as before and gets no new column; the events
table is per directory and format-independent.

---

## 4. Tests and documentation

`tests/test_settings.py`: context collection and read-back; the pattern
process shipping requests; application from `current_t` with old / new
values, override stamping and the next burst; refusals; acknowledgements,
lateness, errors, status and `finish`; the frames table's override
columns; the event log. `tests/test_dry_run.py::test_dry_run_settings_requests_ome_zarr`
runs a method that ramps an exposure and a laser property. The red /
far-red switch example (`documentation/examples/farred_patterns.py`) is
tested with the other examples. Documentation: the "changing settings" section and the
far-red example in `custom_pattern_methods.md`, the events table,
override columns and `status.json` in `data_format.md`, and the
engineering notes.

---

## 5. Decisions (all taken as recommended)

1. Request API: explicit `set_exposure` / `set_config` / `set_property` /
   `set_position` with read-back, rather than one generic call.
2. Position moves: x, y, z and the PFS offset all allowed.
3. Frames table: value in force on every frame from the first change on,
   one column per parameter.
4. Refused requests: warning plus event row; the rest of the request
   applies; nothing raises inside `generate`.
5. Acknowledgements and `status.json` kept in this stage.
