# PyCLM architecture notes

Factual reference for the runtime as of Stage 1 (acquisition plan) on branch
`Stage0-hygiene` (September 2026; the pre-refactor state was `7af035a`). Opinions and recommendations live in
[assessment-2026-09.md](assessment-2026-09.md); bugs live in
[known-issues.md](known-issues.md).

Sizes for orientation: `src/pyclm` is ~5,400 lines. The core pipeline
(`core/manager.py` 900, `core/microscope.py` 270, `core/pattern_process.py` 237,
`core/segmentation_process.py` 176, `core/patterns/pattern.py` 313,
`controller.py` 242, `directories.py` 480) is about 2,600 of those.

---

## 1. Process topology

`Controller` (`controller.py`) builds six process objects, wires them to one
shared `AllQueues`, and runs each one's `process()` method in a
`ThreadPoolExecutor` (`controller.py:164-241`). **They are threads, not OS
processes**, despite the naming. All processes share one `threading.Event`
(`stop_event`) for forced shutdown.

| Process | Class / file | Role | Inherits `BaseProcess`? |
|---|---|---|---|
| Manager | `Manager`, `core/manager.py` | Walks the `AcquisitionPlan`: waits for each timepoint, then turns `plan.events_at(t)` into messages for the other processes. | **No** (own loop) |
| Microscope | `MicroscopeProcess`, `core/microscope.py` | Executes events against a `MicroscopeCoreInterface`: moves stage, sets config groups and device properties, uploads SLM images, snaps. | Yes, but overrides `process()` with its own loop (per-message error guard; aborts after `max_consecutive_errors`, default 10) |
| Outbox | `MicroscopeOutbox`, `core/manager.py` | Hands frames to the configured `FrameWriter` (`core/storage/`: HDF5 format 1 or OME-Zarr format 2) **and** fans them out to Segmentation and Pattern. | Yes (via `DataPassingProcess`) |
| SLM buffer | `SLMBuffer`, `core/manager.py:486` | Holds the latest pattern per experiment, applies the camera→SLM affine, answers the microscope's "give me the current pattern" request. | Yes (via `DataPassingProcess`) |
| Segmentation | `SegmentationProcess`, `core/segmentation_process.py:17` | Runs a `SegmentationMethod` per experiment on frames flagged for segmentation. | Yes |
| Pattern | `PatternProcess`, `core/pattern_process.py:24` | Collects the raw/seg frames a `PatternMethod` declared it needs, runs `generate()`, sends the result to the SLM buffer. | Yes |

`BaseProcess` (`core/base_process.py`) is a poll loop: for each registered
`(queue, handler)` it calls `get_nowait()`, sleeps 1 ms when no queue had
work, catches and logs any handler exception and increments `error_count`
(the process keeps running), and exits when a handler returns `True` or
`stop_event` is set.

### Queues (`core/queues.py`)

All queues are `queue.Queue` (Stage 0 replaced `multiprocessing.Queue`, which
pickled every item through a pipe even between threads). Items are passed by
reference: the object a producer puts is the object the consumer gets, so
nothing downstream may mutate a frame or event it did not create.
`AllQueues.close()` just drains leftovers after every process has exited.

```
manager → microscope     manager_to_microscope     AcquisitionEventMessage, UpdatePatternEventMessage,
                                                   UpdatePositionEventMessage, "close"
manager → outbox         manager_to_outbox         "close"
manager → slm_buffer     manager_to_slm_buffer     UpdatePatternEventMessage, "close"
manager → seg            manager_to_seg            "close"
manager → pattern        manager_to_pattern        RequestPattern, "close"
microscope → manager     microscope_to_manager     UpdateZPositionMessage
microscope → outbox      acquisition_outbox        AcquisitionData | StimulationData, StreamCloseMessage
outbox → seg             outbox_to_seg             AcquisitionData, StreamCloseMessage
seg → outbox             seg_to_outbox             SegmentationData, StreamCloseMessage
outbox → pattern         outbox_to_pattern         AcquisitionData, StreamCloseMessage
seg → pattern            seg_to_pattern            SegmentationData, StreamCloseMessage
pattern → slm_buffer     pattern_to_slm            CameraPattern, StreamCloseMessage
slm_buffer → microscope  slm_to_microscope         EventSLMPattern
```

The topology is fixed by attribute names on `AllQueues`; each process reaches
into `aq.<name>` in its constructor. There is no registry or routing table.

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
    binning, `config_groups`, `devices`, `needs_slm`, and a block of
    **routing/persistence booleans** (`save_output`, `save_stim`, `segment`,
    `save_seg`, `raw_goes_to_pattern`, `seg_goes_to_pattern`) plus method
    names. It is mutated by the microscope after the snap
    (`completed_time`, `pixel_width_um`).
  - `PlannedEvent` (`core/plan.py`) — what the plan hands the Manager: `kind`
    (`request_pattern` / `position` / `update_pattern` / `acquire`), `t`,
    `experiment`, `index`, `scheduled_offset_s`, and the resolved routing
    flags. The Manager's `dispatch()` maps each one onto the messages above.
  - `UpdatePatternEvent` — "put this experiment's current pattern on the SLM".
  - `UpdateStagePositionEvent` — "move to this position".
  - `RequestPattern` (`pattern_process.py`) — a message, not an event: tells
    the pattern process which frames to wait for at an absolute timepoint.
- **Data** (`core/datatypes.py`) carry numpy arrays plus the originating event:
  `AcquisitionData(event, data)`, `StimulationData(event, data, dmd_pattern,
  pattern_id)`, `SegmentationData(event, data)`, `CameraPattern(experiment,
  data, slm_coords, binning)` (gets a fresh `pattern_id` UUID),
  `EventSLMPattern(event_id, pattern, pattern_unique_id)`.

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
  `SegmentationConfig("none")`;
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
   raw, seg)` / `request_stim(raw, seg)` calls into a list of
   `AcquiredImageRequest(channel_id, needs_raw, needs_seg)`. If any request
   needs seg, `SegmentationProcess.request_method(experiment)` constructs the
   `SegmentationMethod` (with shared-resource dedup for e.g. the Cellpose
   model). **Segmentation is never instantiated unless a pattern asks for it**;
   a configured-but-unused segmentation method logs a warning.
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
- The Manager never receives acknowledgements. Its only inbound message is
  `UpdateZPositionMessage` from the microscope (sent when the position mover
  reports a z change > 1 µm), which mutates `self.positions[name].z`
  (`manager.py:718-723`) so later events carry the corrected z. Inbound
  messages are only drained inside the inter-timepoint wait loop.
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
  → acquisition_outbox
MicroscopeOutbox.handle_data
  → write_data(): resize pre-allocated dset, write, write_attrs, flush; dmd + pattern_id for stim;
    bump current_t_index if _timepoint_complete(); a frame with no pre-allocated dataset is
    dropped, logged, and counted in dropped_frames
  → if event.segment:             outbox_to_seg.put(data)
  → if event.raw_goes_to_pattern: outbox_to_pattern.put(data)
SegmentationProcess.handle_segment_data
  → SegmentationData(event, model.segment(img))
  → seg_to_pattern.put(seg)           (always)
  → seg_to_outbox.put(seg)            (if event.save_seg)  → Outbox writes ".../seg"
PatternProcess
  → RequestPattern creates DataDock(time_sec, requirements) keyed (experiment, t)
  → raw/seg arrivals fill the dock (data for a timepoint with no dock is dropped with a warning);
    when complete: docks.pop(), PatternContext(dock, experiment),
    model.generate(context) → CameraPattern(experiment, pattern, slm_coords, binning) → pattern_to_slm
SLMBuffer.handle_data
  → pattern_to_slm(): float[0,1] camera coords → uint8, warpAffine with affine (scaled by binning) → SLM shape
  → slm_patterns[experiment] = (pattern_id, slm_image)
SLMBuffer on UpdatePatternEvent → EventSLMPattern(event.id, slm_image, pattern_id) → slm_to_microscope
MicroscopeProcess.handle_update_pattern_event → core.setSLMImage(); remembers pattern + id for the next StimulationData
```

Who decides routing: `AcquisitionPlan._routing()` resolves the pattern's
`AcquiredImageRequest`s to channel names and sets `segment`, `save_seg`,
`raw_to_pattern`, `seg_to_pattern` on the `PlannedEvent`, but only when a
pattern is being generated this timepoint; `Manager.dispatch()` copies them
onto the `AcquisitionEvent`. The Outbox and Segmentation processes just read
those flags.

What a pattern method sees: `PatternContext` (`patterns/pattern.py:92`) exposes
`.time` (seconds since start, scheduled), `.raw(name)`, `.segmentation(name)`,
`.stim_raw()`, `.stim_seg()` for **the current timepoint only**. The dock is
popped and discarded after `generate()`. Any temporal state must live on the
method instance (e.g. `BounceModel.down`, `EmbryoSegmentationMethod.cached_result`).

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
`segmentation.save` is set. `patterns/dmd/0` is `(N, H_slm, W_slm)` uint8 with
one entry per distinct `pattern_id` (`pattern_policy = on_change`; `all` keeps
one per stimulation event). Root attrs `pyclm`: format, plan YAML, experiment
and schedule metadata, affine transform, SLM shape, groups, `current_t`.
`frames.parquet` (rewritten on every write) and `frames.csv` (at close) in
the experiment directory hold one row per frame and per stimulation event
(`kind`, `t`, group, `local_index`, channel, timestamps, position,
`pattern_id`, `pattern_index`). Skipped timepoints are never written.

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

1. Manager finishes its loop, sends `"close"` to all five manager→X queues,
   returns. `Controller.run` sees the manager future complete and falls through
   to `finally`, which waits for all futures (it does **not** set `stop_event`).
2. Microscope: on `"close"` puts `StreamCloseMessage` on `acquisition_outbox`
   and returns.
3. Outbox: counts `stream_close` messages. First one (from microscope) is
   forwarded to seg and pattern. Exits when `manager_done and stream_count >= 2`
   (microscope + segmentation), closing its files.
4. Segmentation: on `stream_close` forwards to pattern and outbox, exits.
5. Pattern: exits when it has seen `stream_close` from both `from_raw` and
   `from_seg` (`stream_count >= 2`), forwarding one to the SLM buffer.
6. SLM buffer: exits when `manager_done and pattern_done`.

The counts (`>= 2`) are hard-coded to the current topology. Forced shutdown
(`stop_event`, set on `KeyboardInterrupt` or any crashed process) makes every
`BaseProcess` loop break at its next iteration. `MicroscopeOutbox.process`
wraps the loop in `try/finally: close_files()`, and `Controller.run` calls
`close_files()` again in its own `finally`, so files are closed on both paths
(every write is also followed by `flush()`).

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

## 10. Extension points that exist today

- `PatternMethod` subclass + `Controller.register_pattern_method(name, cls)` or
  `run_pyclm(pattern_methods={...})`. Constructor gets only the TOML kwargs;
  hardware context arrives later via `configure_system`. Built-ins are
  registered in `core/patterns/__init__.py:known_models`; each
  `PatternProcess` / `SegmentationProcess` instance copies its registry at
  construction, so registrations do not leak between controllers.
- `SegmentationMethod` subclass (`segment(img) -> labels`), optional
  `request_resource()` for shared heavy models.
- `PositionMover` subclass (`move_to(position, core) -> (z_moved, z)`).
  `PFSPositionMover` raises `TimeoutError` if focus does not lock within
  `PFS_TIMEOUT_S` (30 s).
- `MicroscopeCoreInterface` (Protocol, `core/core_interface.py`) with
  `RealMicroscopeCore` (pymmcore-plus delegate) and `SimulatedMicroscopeCore`.
- Image sources for dry runs: `TimeSeriesImageSource` (folder/yaml, mapping,
  single stack).

## 11. Tests

`uv run --group test pytest` — 114 tests, ~80 s, all passing after Stage 2
(2026-09-06). The dry-run integration tests take almost all of that time.

| File | Covers |
|---|---|
| `test_dry_run.py` | Whole pipeline against the simulated core for each position-list / discovery mode; HDF5 dataset inventory, shapes, dtypes. |
| `test_swmr.py` | Outbox init + write + `convert_hdf5s.make_tif` (HDF5 format 1). |
| `test_storage.py` | Cadence grouping, the OME-Zarr writer end to end (layout, NGFF attrs, compact T, labels, pattern policies, chunks only for acquired frames, frames table), `pyclm.io` readers for both formats, ImageJ export, outbox delegating to the writer. |
| `test_base_process.py` | Poll loop, stop paths, handler error counting. |
| `test_plan.py` | `AcquisitionPlan` enumeration against the scheduling rules over 54 `every_t`/`t_delay`/`t_stop`/stim-cadence combinations, event order and offsets, routing flags, YAML round trip, index → path, stimulation naming, PFS offset recorded not executed, z-readiness of the structure, timing budget, validation. |
| `test_manager_scheduling.py` | Which messages the Manager emits at which `t` from a plan, request/event index agreement, per-timepoint ordering, close fan-out, z-update handling, wait loop not spinning, stop event. |
| `test_pattern_process.py` | Dock keyed by absolute `t`, unrequested data dropped with a warning, per-instance registries. |
| `test_microscope_process.py` | Frame delivery, settle time, error guard and abort threshold, SLM handshake (stale replies, timeout), z-correction message. |
| `test_shutdown.py` | Graceful drain of the five workers after `CloseMessage`; forced stop; HDF5 files closed on both paths. |
| `test_logging_setup.py` | Per-run log handlers. |
| `test_controller_init.py` | Early `FileExistsError`, unused-segmentation warning, settle-time plumbing. |
| `test_pattern_method.py` | Integral `pattern_shape` under binning; `PatternReview` constructible from TOML kwargs. |

`tests/helpers.py` holds the builders (`make_experiment`, `make_schedule`,
`make_plan`, `FakeImageSource`, `drain`) used by the unit tests. Still untested: `SLMBuffer`
transforms, `DataDock` completeness with mixed raw/seg requirements, the
built-in pattern methods (exercised only by the docs zoo).
