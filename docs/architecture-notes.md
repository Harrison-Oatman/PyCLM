# PyCLM architecture notes

Factual reference for the runtime as of Stage 3 (router, registration,
pattern history, tracking) on branch `Stage3-router` (September 2026; the
pre-refactor state was `7af035a`). Opinions and recommendations live in
[assessment-2026-09.md](assessment-2026-09.md); bugs live in
[known-issues.md](known-issues.md).

Sizes for orientation: `src/pyclm` is ~9,300 lines. The core pipeline
(`core/*.py`, `controller.py`, `directories.py`) is about 4,200 of those;
`core/storage/`, `core/patterns/`, `core/tracking/` and `pyclm.io` make up
most of the rest.

---

## 1. Process topology

`Controller` (`controller.py`) builds the process objects, wires the
addressed control queues (`AllQueues`) and the frame-routing table
(`Router`, `core/router.py`), and runs each active process's `process()`
method in a `ThreadPoolExecutor` (`Controller.run`). **They are threads, not
OS processes**, despite the naming. All processes share one
`threading.Event` (`stop_event`) for forced shutdown.

| Process | Class / file | Role | Router role |
|---|---|---|---|
| Manager | `Manager`, `core/manager.py` | Walks the `AcquisitionPlan`: waits for each timepoint, then turns `plan.events_at(t)` into messages for the microscope, the SLM buffer and the pattern process. Between timepoints it applies the setting changes pattern methods asked for (`apply_settings`, `core/settings.py`), absorbs the microscope's acknowledgements (lateness, errors) and writes `status.json`; `finish()` closes the events table. Own loop, not a `BaseProcess`. | none (control only) |
| Microscope | `MicroscopeProcess`, `core/microscope.py` | Executes events against a `MicroscopeCoreInterface`: moves stage, sets config groups and device properties, uploads SLM images, snaps. Own loop with a per-message error guard (aborts after `max_consecutive_errors`, default 10). | produces `raw`; ends the raw stream on the Manager's close |
| Writer | `WriterProcess`, `core/writer_process.py` (`MicroscopeOutbox` is an alias) | Hands every frame, label image and track table to the configured `FrameWriter` (`core/storage/`: OME-Zarr format 2, the default, or HDF5 format 1). | consumes `raw` for every channel and, with `demand=False`, `seg` / `tracks` where the method has `save = true`; `always_active` |
| SLM buffer | `SLMBuffer`, `core/manager.py` | Holds the latest pattern per experiment, applies the camera→SLM affine, answers the microscope's "give me the current pattern" request. | none (the pattern → SLM → microscope path is a control handshake, not routed) |
| Segmentation | `SegmentationProcess`, `core/segmentation_process.py` | Runs one `SegmentationMethod` per `[segmentation]` table the experiment configures (the default table and any `[segmentation.<name>]`) on the frames the router delivers, each at the cadence its consumers need (`segmentations_for`). | produces `seg` and `seg:<name>` from `raw`; started only where some segmentation is demanded |
| Tracking | `TrackingProcess`, `core/tracking_process.py` | Runs a `TrackingMethod` per (experiment, channel) on every segmentation of that channel. | produces `tracks` from `seg`; `continuous` (needs every frame); started only where `tracks` is demanded |
| Pattern | `PatternProcess`, `core/pattern_process.py` | Collects the raw/seg/tracks data a `PatternMethod` declared it needs, keeps a bounded per-experiment history, runs `generate()`, sends the result to the SLM buffer. | consumes at `"pattern"` cadence; `always_active` |

`BaseProcess` (`core/base_process.py`) is a poll loop: for each registered
`(queue, handler)` it calls `get_nowait()`, sleeps 1 ms when no queue had
work, catches and logs any handler exception and increments `error_count`
(the process keeps running), and exits when a handler returns `True` or
`stop_event` is set. `PipelineProcess` (same file) is the router-facing
subclass: it declares `produces` (`{kind: (input kinds,)}`), `continuous`,
`always_active`, `subscriptions(plan)` and `can_produce(kind, experiment,
channel)`; the router hands it one data inbox (`attach`), on which data and
the final `StreamCloseMessage` arrive in order; `handle_data(data)` is the
only method a consumer implements and `publish(data)` is how a producer
emits. `Controller.add_process(proc)` registers an extra `PipelineProcess`
before `initialize()`.

### Control queues (`core/queues.py`)

All queues are `queue.Queue` (Stage 0 replaced `multiprocessing.Queue`, which
pickled every item through a pipe even between threads). Items are passed by
reference: the object a producer puts is the object the consumer gets, so
nothing downstream may mutate a frame or event it did not create.
`AllQueues.close()` just drains leftovers after every process has exited.

```
manager → microscope     manager_to_microscope     AcquisitionEventMessage, UpdatePatternEventMessage,
                                                   UpdatePositionEventMessage, "close"
manager → slm_buffer     manager_to_slm_buffer     UpdatePatternEventMessage, "close"
manager → pattern        manager_to_pattern        RequestPattern, "close"
microscope → manager     microscope_to_manager     UpdateZPositionMessage, EventDoneMessage (one per acquisition)
pattern → manager        pattern_to_manager        SettingsRequestMessage (a method's setting changes)
pattern → slm_buffer     pattern_to_slm            CameraPattern, StreamCloseMessage
slm_buffer → microscope  slm_to_microscope         EventSLMPattern
```

### The router (`core/router.py`)

Frame-derived data (`AcquisitionData` / `StimulationData` = kind `raw`,
`SegmentationData` = `seg`, or `seg:<name>` for a named
`[segmentation.<name>]` table, `TrackingData` = `tracks`; `core/kinds.py`
holds the vocabulary) does not travel on named queues. `Controller.initialize` builds a `Router(plan)`, `add()`s the
microscope, writer, segmentation, tracking, pattern and any user process,
and calls `resolve()`, which:

1. collects each process's `subscriptions(plan)` (`Subscription(consumer,
   experiment, channel, kind, cadence, demand)`; cadence `"always"` or
   `"pattern"` = only at timepoints where `plan.pattern_due(experiment, t)`);
2. for every demanded kind other than `raw`, subscribes the kind's producer
   to its input kinds at the widest downstream cadence (`"always"` if the
   producer is `continuous`), iterating until nothing changes; a demanded
   kind with no producer, or whose producer's `can_produce` says the
   experiment has no method configured, raises `RoutingError`;
3. adds `demand=False` subscriptions (the writer's) only for data that is
   produced anyway;
4. computes each consumer's upstream set (the producers of the kinds it
   receives, plus the raw producer), creates one inbox per active process
   (something routed to it, or `always_active`) and calls `attach`.

`publish(data)` looks up `(event.experiment_name, event.index["c"],
data.kind)` and puts the same object on each subscriber's inbox whose
cadence is due at `event.t_index`; a key nobody subscribes to is counted in
`undeliverable` and logged once. `end_stream(producer)` puts one
`StreamCloseMessage` on a consumer's inbox when the last of its upstreams
has ended. `as_dict()` is the resolved table; it is logged at INFO and
stored in the outputs (`pyclm.routing` in the zarr root attrs, `routing`
attribute in HDF5).

The router is not a thread: `publish` runs on the producer's thread and
costs one `put` per delivery, so a frame reaches segmentation before the
writer has compressed it.

---

## 2. Vocabulary: messages, events, data

- **Messages** (`core/messages.py`) carry a string tag in `.message`; every
  process dispatches with `match msg.message:`. Most wrap an event.
  `CloseMessage` (`"close"`) is what the Manager sends when the schedule ends.
- **Events** (`core/events.py`) describe *intent*:
  - `AcquisitionEvent` — one snap. Carries experiment name, a
    `MicroscopePosition`, `channel_id` (UUID of the `ImagingConfig`),
    scheduled wall time, `index` (`{"t", "p", "c"}`; `t_index` is a property
    on it, and `get_rel_path()` derives the HDF5 path from it), exposure,
    binning, `config_groups`, `devices`, `needs_slm` and `save_output`.
    Nothing about who consumes the frame (Stage 3 removed the routing
    booleans and method names). It is mutated by the microscope after the
    snap (`completed_time`, `pixel_width_um`).
  - `PlannedEvent` (`core/plan.py`) — what the plan hands the Manager: `kind`
    (`request_pattern` / `position` / `update_pattern` / `acquire`), `t`,
    `experiment`, `index`, `scheduled_offset_s`, `channel`, `is_stim`, `save`.
    The Manager's `dispatch()` maps each one onto the messages above.
  - `UpdatePatternEvent` — "put this experiment's current pattern on the SLM".
  - `UpdateStagePositionEvent` — "move to this position".
  - `RequestPattern` (`pattern_process.py`) — a message, not an event: tells
    the pattern process which frames to wait for at an absolute timepoint.
- **Data** (`core/datatypes.py`) carry numpy arrays plus the originating
  event, and a class-level `kind` the router keys on: `AcquisitionData(event,
  data)` (`raw`), `StimulationData(event, data, dmd_pattern, pattern_id)`
  (`raw`), `SegmentationData(event, data, name)` (`seg`, or `seg:<name>`
  for a named table; the router looks producers up by the base kind, so one
  segmentation process serves every name), `TrackingData(event,
  labels, rows)` (`tracks`; `rows` are `TrackRow`s). Not routed:
  `CameraPattern(experiment, data, slm_coords, binning)` (gets a fresh
  `pattern_id` UUID) and `EventSLMPattern(event_id, pattern,
  pattern_unique_id)`.

---

## 3. Configuration → schedule → plan

`directories.py:schedule_from_directory()` builds an `ExperimentSchedule`
(`core/experiments.py`) from an experiment directory, and
`AcquisitionPlan.from_schedule()` (`core/plan.py`) turns that into the plan
the pipeline runs from (see the end of this section).

1. Every `*.toml` in the directory is a candidate experiment config, keyed by
   file stem.
2. Positions come from `PositionList.pos` (MicroManager JSON; `-` in labels
   is replaced by `.`) or, as fallback, `multipoints.xml` (Nikon Elements).
3. Each position label is split on `.`; the first token must match a TOML
   stem. **One position = one `Experiment` = one HDF5 file.** The dict keys of
   `schedule.experiments` and `schedule.positions` are the full position
   labels (e.g. `bar10.00`), and `ExperimentSchedule.__init__` asserts the two
   key sets are equal.
4. `schedule.toml` `[timing]` → `TimeCourse(count, interval, setup, between)`.

`experiment_from_toml()` (`directories.py:24`) builds, per experiment:

- a base `ImagingConfig` from top-level `[config_groups]` / `[device_properties]`;
- an imaging default from `[imaging]` (exposure, every_t, save, binning);
- one `ImagingConfig` per preset in `[channels].presets`, each with
  `ConfigGroup(channels.group, preset)` added and optional per-preset
  overrides (exposure, every_t, config_groups, device_properties; binning
  override is deliberately disabled);
- a stimulation `ImagingConfig` from `[stimulation]` (its `binning` defaults
  to the imaging binning);
- `SegmentationConfig(method, **rest)` from `[segmentation]` or
  `SegmentationConfig("none")`, plus one per `[segmentation.<name>]`
  sub-table, all in `Experiment.segmentations` (`experiment.segmentation`
  is the default entry, `segmentation_names` the configured ones);
- `TrackingConfig(method, segmentation=..., **rest)` from `[tracking]`
  (`segmentation` names the table the tracker links);
- `PatternConfig(method, **rest)` from `[pattern]` (`every_t` is read from
  the kwargs by `MethodBasedConfig`);
- `t_delay`, `t_stop` (in timepoints).

`ImagingConfig` instances each get a UUID `channel_id`; this is the join key
between events, data, pattern requirements, and docks. There is no schema
validation; missing keys surface as `KeyError`, unknown keys are passed on as
method kwargs.

### The acquisition plan (`core/plan.py`)

`AcquisitionPlan` is the single place scheduling logic lives. It holds a
`useq.MDASequence` (time plan; one `Position` per experiment with a
sub-sequence listing its channels, stimulation first with `do_stack=False`;
an `AxesBasedAF` carrying the PFS offset for provenance only) plus the
`ExperimentSchedule` for hardware detail. Everything useq cannot express is
under `metadata["pyclm"]`: `t_delay`, `t_stop`, per-channel `every_t`,
`pattern_every_t`, the stimulation channel's name (its preset in the channel
group, e.g. `DMD`, else `stimulation`), binning, and the pattern's
requirements. The wrapper, not useq's iterator, decides *when*: cadence is
experiment-relative (`(t - t_delay) % every_t == 0`), and
`Channel.acquire_every` is deliberately left at 1.

Queries: `events_at(t)` (ordered `PlannedEvent`s: per active experiment a
`request_pattern` when `(t - t_delay) % lcm == 0`, a `position`, then
`update_pattern` + `acquire` for stimulation and `acquire` per imaging
channel; nothing when no frame is due), `is_scheduled(experiment, channel,
t)`, `datasets_at()` / `expected_datasets()`, `imaging_config()`,
`estimate_timepoint_s()` / `over_budget()`, `to_yaml()` / `from_yaml()`.
`structure(experiment)` exposes useq's own per-timepoint events for a
position, which is where z and grid axes will appear.

Dry runs use `dry_schedule_from_directory()`, which additionally resolves a
TIF per position (priority: `dry_run.yml`, then position list + TIF name
matching, then TIF file names as positions) and builds a
`TimeSeriesImageSource` keyed by `(x, y)`.

---

## 4. Startup sequence (`Controller.initialize`, `controller.py`)

0. Raise `FileExistsError` if any `<experiment>.hdf5` already exists in the
   output directory, before any model is loaded.
1. If simulated, push `slm_shape` into the core.
2. Set binning 1 on the camera, read ROI and pixel size → `CameraProperties`.
3. `PatternProcess.initialize(camera_properties)`.
4. For each experiment: `PatternProcess.request_method(experiment)` constructs
   the `PatternMethod` from `pattern.kwargs` **only** and calls
   `model.initialize(experiment)`, which resolves `add_requirement(channel_name,
   raw, seg, tracks, history)` (`seg` may name `[segmentation.<name>]`
   tables) / `request_stim(raw, seg, history)` calls into a list of
   `AcquiredImageRequest(channel_id, needs_raw, needs_seg, needs_tracks,
   history, segmentations)`, whose `.kinds` are the routing kinds. After the
   router resolves (step 7), `Controller._instantiate_producers` calls
   `SegmentationProcess.request_method(experiment, name)` for every demanded
   `(experiment, segmentation name)` (with shared-resource dedup for e.g. the
   Cellpose model) and `TrackingProcess.request_method` per tracked channel.
   **A segmentation is never instantiated unless something asks for it**; a
   configured-but-unused table logs a warning.
5. `t_gcd` of all channel `every_t` values is computed for the GUI.
6. `PatternProcess.initialize_models()` → `configure_system()` on each method
   injects camera properties and the experiment reference and applies the
   stimulation binning to `pattern_shape` / `pixel_size_um`.
7. `AcquisitionPlan.from_schedule(schedule, requirements)` is built and
   written to `plan.useq.yaml` in the experiment directory; if
   `plan.over_budget(settle_s)` finds timepoints whose estimated duration
   exceeds the interval, a warning names the worst one. `Manager.initialize(plan)`.
8. `SLMBuffer.initialize(slm_shape, affine, names)` → blank pattern per
   experiment; `MicroscopeProcess.declare_slm()`.
9. `MicroscopeOutbox.initialize(plan, core)` creates every HDF5 file, embeds
   the plan YAML in its root attributes, and **pre-allocates one dataset per
   `plan.expected_datasets()` entry** (see §7), then enables SWMR. Returns the `(file, channel)` layer list that is written to
   `all_layers.txt` (JSON) for the GUI subprocess.

`run_pyclm()` (`run_pyclm.py`) is the front door: finds `pyclm_config.toml`
(experiment dir, then CWD), sets up per-run logging (`set_logging` replaces
the handlers from any previous run in the same interpreter), reads the
optional `focus_device` (default `"ZDrive"`) and `settle_time_seconds`
(default 1.0) keys, builds the schedule, constructs `Controller`, registers
custom methods, selects the focus device, calls `initialize`, optionally
spawns the GUI as a **separate OS process** (`subprocess.Popen` of
`pyclm.gui.gui_controller`), and calls `Controller.run()`.

---

## 5. Timing model (`Manager.process`, `manager.py:778`)

```
start = now + plan.setup_s
for t in range(plan.timepoints):
    wait until (now - start) >= plan.time_offset_s(t) - setup   # drains microscope_to_manager, sleeps 10 ms
    for ev in plan.events_at(t):                                # per active experiment, in order:
        dispatch(ev)                                            #   request_pattern? position,
                                                                #   [update_pattern, acquire(stim)], acquire(channel)...
send CloseMessage to all five outboxes
```

`plan.events_at(t)` applies the rules that used to be inline here: skip an
experiment when `t < t_delay` or past `t_stop`; a channel (or stimulation) is
acquired when `(t - t_delay) % every_t == 0`; a pattern request is sent when
`(t - t_delay) % lcm(pattern.every_t, every_t of the required channels) == 0`,
and only then do the required channels' events carry routing flags; a
position event is sent only when at least one frame is acquired. Each
event's `scheduled_offset_s` is `t * interval + p_index * between`.

Key properties:

- All events for a timepoint are emitted in one burst at the *preparatory*
  phase, `setup` seconds before the timepoint. The microscope then paces
  itself: for each `AcquisitionEvent` it sleeps until `scheduled - 0.1 s`,
  calls `waitForSystem()`, sleeps `settle_time_s` (from
  `pyclm_config.toml: settle_time_seconds`, default 1.0), then snaps.
  Channels of one experiment share one `scheduled` time, so they run back to
  back.
- `between` is a per-experiment offset. Nothing checks whether experiment *i*
  finished before experiment *i+1*'s scheduled time; overruns simply run late.
  The `todo: check if we are behind schedule` at `manager.py:791` is still open.
- The Manager receives three inbound messages, all drained only inside the
  inter-timepoint wait loop, which makes that loop the boundary at which
  they take effect: `UpdateZPositionMessage` from the microscope (a z
  change > 1 µm from the position mover; mutates `positions[name].z` and
  is recorded as a `z_correction` event), `SettingsRequestMessage` from
  the pattern process (a method's changes to its own experiment's
  exposure, presets, device properties or position; validated with
  `settings.check_change`, applied to the live `ImagingConfig` /
  `MicroscopePosition` from `current_t`, recorded in `events.parquet`, and
  stamped as `event.overrides` on later acquisition events so the frames
  table gets a column per changed setting), and `EventDoneMessage` from
  the microscope after every acquisition (lateness = completed − scheduled;
  a timepoint more than one interval late is warned about once and
  recorded). `status.json` is written before every burst and by
  `finish()`, which the Controller calls after all processes exit.
- `t_index` on events and on `RequestPattern` is the **absolute** timepoint
  `t`; only the cadence decisions (`% every_t`, `% lcm`) use the
  experiment-relative `this_t`. (Before Stage 0 the request used `this_t`,
  which broke pattern generation for `t_delay > 0`; see known-issues #1.)

### Closed-loop latency

`UpdatePatternEvent` for timepoint *t* is answered by `SLMBuffer` with whatever
pattern it currently holds, i.e. the most recent `generate()` result, which was
computed from frames acquired at an earlier timepoint (typically *t − lcm*).
The microscope stores that pattern and its `pattern_id`, and the stimulation
`StimulationData` carries both to the outbox, which writes the DMD image and
`pattern_id` next to the stimulation frame. So the loop has a built-in
one-interval delay, and a slow segmentation degrades gracefully to "reuse the
stale pattern" rather than blocking; the recorded `pattern_id` lets analysis
recover which pattern was actually applied. Initially the pattern is all zeros.

### The one synchronous handshake

`MicroscopeProcess.handle_update_pattern_event` blocks on `slm_to_microscope`
for up to `slm_await_s` (default 5 s) waiting for the `EventSLMPattern` whose
`event_id` matches. Replies for other events are discarded with a warning; on
timeout it logs a warning and keeps the pattern already on the SLM (whose
`pattern_id` is what the next `StimulationData` records). Any exception in a
handler is logged and counted; after `max_consecutive_errors` in a row the
microscope re-raises and the Controller aborts the run.

---

## 6. Data flow per frame

```
MicroscopeProcess.handle_acquisition_event
  → set config groups, device props, exposure, binning; wait; snap
  → AcquisitionData(event, img)  or  StimulationData(event, img, current_pattern, pattern_id)
  → router.publish(data)                       key (experiment, channel, "raw")
Router.publish
  → for each subscriber of the key whose cadence is due at event.t_index: inbox.put(data)
    (the writer always; segmentation if seg is demanded for the channel; the pattern
    process if it asked for the raw frame; a user process if it subscribed)
WriterProcess.handle_data
  → FrameWriter.write_frame / write_labels / write_tracks by data.kind
    (OME-Zarr: chunk write, frames/tracks table rows, current_t; HDF5 v1: resize the
    pre-allocated dataset, write attrs, flush, current_t_index; a frame with no slot is
    dropped, logged, and counted in dropped_frames)
SegmentationProcess.handle_data
  → for each (segmentation name, cadence) the router wants of this channel, skipping
    "pattern"-cadence ones when the pattern is not due:
    SegmentationData(event, model.segment(img), name)    → router.publish   "seg" / "seg:<name>"
TrackingProcess.handle_data                              (only if tracks are demanded; every frame)
  → labels, rows = method.track(seg, t, pixel_size_um)
  → TrackingData(event, labels, rows)                    → router.publish   "tracks"
PatternProcess
  → RequestPattern creates DataDock(time_sec, requirements) keyed (experiment, t)
  → raw/seg/tracks arrivals fill the dock by data.kind (data for a timepoint with no dock is
    dropped with a warning); when complete: docks.pop(), ExperimentState.absorb(dock, t),
    PatternContext(state, experiment), model.generate(context)
    → CameraPattern(experiment, pattern, slm_coords, binning) → state.record_pattern → pattern_to_slm
SLMBuffer.handle_data
  → pattern_to_slm(): float[0,1] camera coords → uint8, warpAffine with affine (scaled by binning) → SLM shape
  → slm_patterns[experiment] = (pattern_id, slm_image)
SLMBuffer on UpdatePatternEvent → EventSLMPattern(event.id, slm_image, pattern_id) → slm_to_microscope
MicroscopeProcess.handle_update_pattern_event → core.setSLMImage(); remembers pattern + id for the next StimulationData
```

Who decides routing: the `Router` (§1), from the pattern method's
`AcquiredImageRequest`s (`plan.pattern_requirements()`), the writer's
recording subscriptions, and the producers' declarations. Deliveries to the
pattern process happen only at timepoints where a pattern is due
(`plan.pattern_due`), which is also when the Manager sends `RequestPattern`;
deliveries to tracking happen at every frame. Nothing is decided per event.

What a pattern method sees: `PatternContext` (`patterns/pattern.py`) on the
experiment's `ExperimentState`: `.t`, `.time`, `.generation`, `.raw(name)`,
`.segmentation(channel, name=)` (a named table's labels), `.regions(channel,
name=)` (a `Regions`), `.tracks(channel)` (a `Tracks`, itself a `Regions`:
relabelled mask, rows, `mask(id)`, `centroid(id)`, `measure`, `paint`),
`.stim_raw()`, `.stim_seg()`, `.history(name,
kind, n)` (the last `history` deliveries declared in `add_requirement`, at
the pattern's cadence), `.stim_history()`, `.last_pattern()`,
`.pattern_history(n)`. The dock is still popped after each timepoint; the
state persists for the run and is bounded by the declared depths (default 1)
and `PatternMethod.pattern_history` (default 2). `ZooContext` mirrors the
same methods for the docs zoo.

---

## 7. Storage layouts

Selected by `[output] format` in `pyclm_config.toml` (`hdf5`, the default, or
`ome-zarr`). `MicroscopeOutbox` owns a `FrameWriter` (`core/storage/base.py`)
with `open(plan, core, base_path, affine, slm_shape)`, `write_frame`,
`write_labels`, `close`. Both writers read the same plan; `pyclm.io.open()`
reads both layouts behind one API (`ExperimentData`, `GroupData`), and
`pyclm.io.export_imagej()` writes ImageJ hyperstacks from either.

### OME-Zarr (format 2, `core/storage/ome_zarr.py`)

One NGFF 0.4 / zarr v2 store per experiment, `<experiment>.zarr/`. Channels
are grouped by cadence (`cadence_groups()`): each group is an NGFF image
`<group>/0` shaped `(T, C, Y, X)` uint16 with a **compact** time axis (one
slot per acquisition of that group; `every_t` and `t_delay` in the group's
`pyclm` attrs and as the NGFF time-scale transform), chunks `(1, 1, Y, X)`,
zstd. In the common configuration there is one group, `imaging`; a saved
stimulation frame joins the group of its cadence (last channel) or forms
`stim`. `<group>/labels/segmentation/0` holds label images when
`segmentation.save` is set, and `<group>/labels/<name>/0` those of each named
`[segmentation.<name>]` table the writer records (the NGFF `labels` list
names them, default first). `patterns/dmd/0` is `(N, H_slm, W_slm)` uint8 with
one entry per distinct `pattern_id` (`pattern_policy = on_change`; `all` keeps
one per stimulation event). Root attrs `pyclm`: format, plan YAML, experiment
and schedule metadata, affine transform, SLM shape, groups, `current_t`.
`frames.parquet` (rewritten on every write) and `frames.csv` (at close) in
the experiment directory hold one row per frame and per stimulation event
(`kind`, `t`, group, `local_index`, channel, timestamps, position,
`pattern_id`, `pattern_index`). Skipped timepoints are never written.

Tracks (Stage 3): `<group>/labels/tracks/0` `(T, C, Y, X) uint32` next to
`labels/segmentation`, created for the channels whose tracks the writer will
receive (`recorded`, from the router), and `tracks.parquet` / `tracks.csv`
in the experiment directory (columns `experiment, t, channel, group,
local_index, track_id, label, y, x, y_um, x_um, area, parent`). The resolved
routing table is stored under `pyclm.routing` in the root attrs (HDF5 v1: a
`routing` root attribute; v1 does not store tracks and warns once).

### HDF5 (format 1, `core/storage/hdf5_v1.py`, one file per experiment/position)

Root attributes: `schedule_metadata` (JSON of `ExperimentSchedule.as_dict()`),
`experiment_metadata` (JSON of `Experiment.as_dict()`), `plan` (the
`plan.useq.yaml` text), `plan_format` (1), `every_t` (JSON map
`channel_<name>`/`stim_aq` → int), `t_delay`, `t_stop`, `t_count`.

Root dataset `current_t_index` (int32 scalar, starts −1): the highest
timepoint for which every *saved* scheduled dataset has been written. The GUI
polls it.

Per timepoint group `{t:05d}` (absolute `t`, only for timepoints where the
experiment is active):

```
{t:05d}/stim_aq/data   uint16, maxshape = camera ROI // stim binning   (only if stim.exposure > 0 and this_t % stim.every_t == 0)
{t:05d}/stim_aq/seg    uint16 (only if segmentation.save; note "none" seg has save=True)
{t:05d}/stim_aq/dmd    uint8,  maxshape = SLM shape (only if an SLM device exists)
{t:05d}/channel_<name>/data   uint16, maxshape = camera ROI // channel binning
{t:05d}/channel_<name>/seg    uint16 (same condition as above)
```

All datasets are created with shape `(0, 0)` and chunks before `swmr_mode =
True`, and resized to the actual frame shape on write. Never-written datasets
stay `(0, 0)`; readers treat that as "absent". Each dataset has a fixed set of
attributes pre-created by `_preallocate_attrs` and overwritten by
`AcquisitionEvent.write_attrs` (`as_attrs()`), including `index` (JSON of the
frame index); `dmd` also gets `pattern_id`. Position is stored as a list of
`(key, str(value))` pairs.

Why pre-allocate: SWMR readers are only guaranteed to see objects that existed
when they opened the file. (Empirically, h5py 3.14 / HDF5 1.14.6 does *not*
raise when creating attributes or datasets after `swmr_mode = True`; the
constraint is about reader visibility, not writer errors.)

Consumers of this layout: `MicroscopeOutbox._timepoint_complete`, the GUI
(`gui/gui_controller.py`), `convert_hdf5s.py`, `PatternReview`, `tests/test_dry_run.py`.

Other outputs in the experiment directory: `plan.useq.yaml` (the acquisition
plan, also embedded in every output), `<experiment>_<group>.tif` ImageJ
hyperstacks when `[output] export_imagej` is on (default), `log.log` (file
handler at INFO, console at WARNING), `all_layers.txt` (JSON: `{"t": t_gcd,
"all_layers": ["path:layer", ...]}` where `layer` is `group/channel` for
zarr or `channel_x` / `stim_aq` for HDF5).

---

## 8. Shutdown protocol

Normal completion is a drain, not a stop:

1. Manager finishes its loop, sends `"close"` to its three addressed queues
   (microscope, SLM buffer, pattern), returns. `Controller.run` sees the
   manager future complete and falls through to `finally`, which waits for
   all futures (it does **not** set `stop_event`).
2. Microscope: on `"close"` calls `router.end_stream("microscope")` and
   returns.
3. Router: for every consumer whose upstreams have all ended it puts one
   `StreamCloseMessage` on that consumer's inbox. Upstreams are derived from
   the table at `resolve()`: every consumer waits for the raw producer plus
   each producer that feeds it (`Router.upstreams(name)`), so in the closed
   loop segmentation closes first, then tracking (if any), then the pattern
   process and the writer.
4. A `PipelineProcess` on `StreamCloseMessage` runs `on_stream_end()`: the
   default ends its own stream (which may close consumers downstream) and
   exits; the writer closes its outputs first; the pattern process forwards
   a `StreamCloseMessage` to the SLM buffer first.
5. SLM buffer: exits when `manager_done and pattern_done`.

No process counts anything; adding a consumer adds nothing to shut down.
Forced shutdown (`stop_event`, set on `KeyboardInterrupt` or any crashed
process) makes every `BaseProcess` loop break at its next iteration;
`PipelineProcess.process` ends its stream in a `finally`,
`WriterProcess.process` closes the outputs in a `finally`, and
`Controller.run` calls `close_files()` again in its own `finally`, so outputs
are closed on both paths.

---

## 9. Coordinate frames and units

| Frame | Where it lives | Notes |
|---|---|---|
| Stage µm (x, y, z, extras) | `MicroscopePosition` | `PFSPositionMover` negates y when calling `setXYPosition` (`position_mover.py:82`). |
| Camera pixels, full ROI | `CameraProperties.roi` from `core.getROI()` | `(x_off, y_off, w, h)`; `get_image_shape()` returns `(h//b, w//b)`. |
| Pattern space | `PatternMethod.pattern_shape`, `pixel_size_um` | Camera ROI divided by **stimulation** binning; methods return float `[0,1]` arrays of this shape. `update_binning` keeps `pattern_shape` integral (the `int(h)` casts in built-in methods predate that and are harmless). |
| SLM pixels | `SLMBuffer.slm_shape`, `pyclm_config.affine_transform` (2×3, camera→SLM) | `pattern_to_slm` scales the linear part by binning, converts to uint8 0–255, `cv2.warpAffine`. `PatternMethodReturnsSLM` subclasses skip the transform. |
| Time | `context.time` = scheduled seconds since `start`; `t_index` = absolute timepoint | Pattern methods convert to minutes themselves. |

The simulated core's `getROI()` returns 4× the TIF size (`simulated_core.py:192`)
so that binning-4 configs produce the TIF shape; it never actually bins images.

---

## 10. Extension points

- `PatternMethod` subclass + `Controller.register_pattern_method(name, cls)` or
  `run_pyclm(pattern_methods={...})`. Constructor gets only the TOML kwargs;
  hardware context arrives later via `configure_system`. Requirements:
  `add_requirement(channel, raw, seg, tracks, history)` (`seg`: `True`, a
  `[segmentation.<name>]` name, or a list of names), `request_stim(raw,
  seg, history)`. The measurement toolbox (`core/measure.py`: `Regions`,
  `PerTrack`, `nuclear_cytosolic_ratio`; `Tracks` is a `Regions`) is what
  methods use to measure per object, remember per track and paint
  per-object values back into a pattern. The per-cell bases (`PerCellPatternMethod`,
  `NucleusControlMethod`) take `tracks=True` to run their loop on tracked
  labels and expose `cell_labels(context)` as the hook that supplies the
  label image. Built-ins are registered in
  `core/patterns/__init__.py:known_models`; each process instance copies its
  registry at construction, so registrations do not leak between controllers.
- `SegmentationMethod` subclass (`segment(img) -> labels`), optional
  `request_resource()` for shared heavy models.
- `TrackingMethod` subclass (`track(labels, t, pixel_size_um) -> (relabelled,
  rows)`, `core/tracking/`) + `Controller.register_tracking_method` or
  `run_pyclm(tracking_methods={...})`; one instance per (experiment,
  channel), configured by `[tracking]`. Built-in: `centroid`.
- `PipelineProcess` subclass + `Controller.add_process(proc)`: a new consumer
  or producer declares `produces`, `continuous`, `always_active`,
  `subscriptions(plan)` and implements `handle_data`; the router wires it and
  derives its shutdown. `TrackingProcess` is the reference example.
- `FrameWriter` subclass (`core/storage/base.py`): `open`, `write_frame`,
  `write_labels`, `write_tracks` (optional), `close`, `planned_paths`,
  `output_paths`; selected by `make_writer`.
- `PositionMover` subclass (`move_to(position, core) -> (z_moved, z)`).
  `PFSPositionMover` raises `TimeoutError` if focus does not lock within
  `PFS_TIMEOUT_S` (30 s).
- `MicroscopeCoreInterface` (Protocol, `core/core_interface.py`) with
  `RealMicroscopeCore` (pymmcore-plus delegate) and `SimulatedMicroscopeCore`.
- Image sources for dry runs: `TimeSeriesImageSource` (folder/yaml, mapping,
  single stack).

Concurrency is threads and `queue.Queue` (Stage 0). If a stage ever has to
move out of process, the Router is the one place to do it: a consumer whose
inbox is a `multiprocessing.Queue` and whose `publish` path pickles is a
router concern, invisible to producers and to the Controller.

## 11. Tests

`uv run --group test pytest` — 180 tests, ~115 s, all passing after Stage 4
(2026-09-08). The dry-run integration tests take almost all of that time.

| File | Covers |
|---|---|
| `test_dry_run.py` | Whole pipeline against the simulated core for each position-list / discovery mode (HDF5 dataset inventory), the OME-Zarr run, a closed loop with segmentation + tracking on OME-Zarr (tracked labels, tracks table, routing provenance, export), two named segmentations, and a method changing its stimulation channel's exposure and a laser property mid-run (events table, override columns, status file). |
| `test_router.py` | Table resolution (pattern-only, tracking widens segmentation to every frame, recording only what is produced, shared producers), validation errors, cadence filtering and object identity on publish, undeliverable counting, stream close after the last upstream, exactly once, under concurrent producers. |
| `test_shutdown.py` | Graceful drain of the workers after `CloseMessage` in open loop, with segmentation, and with tracking (derived upstreams); forced stop; outputs closed on both paths. |
| `test_doc_examples.py` | The pattern methods shown in the user docs (`documentation/examples/`: leader cells, three-phase intensity programme on the toolbox, the KTR clamp on two named segmentations, the red / far-red switch that turns lasers on and off from a programme string) against synthetic data, and the `tracks = true` switch on the per-cell base classes. |
| `test_settings.py` | Runtime setting changes: the context collecting and reading back settings, the pattern process shipping requests, the Manager applying them from `current_t` (old / new values, override stamping, the next burst), refusals, acknowledgements with lateness and errors, `status.json`, `finish()`, the frames table's override columns, the event log. |
| `test_measure.py` | The measurement toolbox: `Regions` (ids, areas, centroids, `measure` statistics, `paint` from scalar / dict / array, `select`, `owner_of`), `Tracks` as a `Regions` in row order, `PerTrack` defaults, `nuclear_cytosolic_ratio`. |
| `test_named_segmentation.py` | The `seg:<name>` vocabulary, `[segmentation.<name>]` parsing and `Experiment.segmentations`, requirements naming segmentations, dock slots and context accessors per name, the router serving two segmentations of one channel at their own cadences (record-only subscribers never widen production), tracking a named table, a missing named table as a `RoutingError`, the segmentation process without a router. |
| `test_tracking.py` | The centroid linker (id stability, new ids, µm gate, empty frames), `Tracks`, `TrackingProcess`, `[tracking]` parsing, router wiring for tracks, `context.tracks()`. |
| `test_storage.py` | Cadence grouping, the OME-Zarr writer end to end (layout, NGFF attrs, compact T, labels, tracks, pattern policies, chunks only for acquired frames, frames and tracks tables, routing attrs), `pyclm.io` readers for both formats, ImageJ export, the writer process delegating to the writer, HDF5 v1 dropping tracks with one warning. |
| `test_swmr.py` | Writer init + write + `convert_hdf5s.make_tif` (HDF5 format 1). |
| `test_base_process.py` | Poll loop, stop paths, handler error counting. |
| `test_plan.py` | `AcquisitionPlan` enumeration against the scheduling rules over 54 `every_t`/`t_delay`/`t_stop`/stim-cadence combinations, event order and offsets, pattern requirements and cadence, YAML round trip, index → path, stimulation naming, PFS offset recorded not executed, z-readiness of the structure, timing budget, validation. |
| `test_manager_scheduling.py` | Which messages the Manager emits at which `t` from a plan, request/event index agreement, per-timepoint ordering, close fan-out to the addressed queues, z-update handling, wait loop not spinning, stop event. |
| `test_pattern_process.py` | Dock keyed by absolute `t`, unrequested data dropped with a warning, per-instance registries, subscriptions at pattern cadence, history depth and order, previous patterns, unrequested history rejected. |
| `test_microscope_process.py` | Frame delivery to the router, settle time, error guard and abort threshold, SLM handshake (stale replies, timeout), z-correction message. |
| `test_logging_setup.py` | Per-run log handlers. |
| `test_controller_init.py` | Early `FileExistsError` for both formats, unused-segmentation warning, settle-time plumbing. |
| `test_pattern_method.py` | Integral `pattern_shape` under binning; `PatternReview` constructible from TOML kwargs. |

`tests/helpers.py` holds the builders (`make_experiment`, `make_schedule`,
`make_plan`, `FakeImageSource`, `drain`) used by the unit tests. Still
untested: `SLMBuffer` transforms, the built-in pattern methods (exercised
only by the docs zoo).
