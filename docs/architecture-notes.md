# PyCLM architecture notes

Factual reference for the runtime as it exists on branch `core-refactor`
(HEAD `7af035a`, September 2026). Opinions and recommendations live in
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
| Manager | `Manager`, `core/manager.py:620` | Walks the schedule, emits acquisition / SLM / position events on a timer, requests patterns. | **No** (own loop) |
| Microscope | `MicroscopeProcess`, `core/microscope.py:22` | Executes events against a `MicroscopeCoreInterface`: moves stage, sets config groups and device properties, uploads SLM images, snaps. | Yes, but overrides `process()` with its own loop |
| Outbox | `MicroscopeOutbox`, `core/manager.py:124` | Writes frames to per-experiment HDF5 files (SWMR) **and** fans frames out to Segmentation and Pattern. | Yes (via `DataPassingProcess`) |
| SLM buffer | `SLMBuffer`, `core/manager.py:486` | Holds the latest pattern per experiment, applies the camera→SLM affine, answers the microscope's "give me the current pattern" request. | Yes (via `DataPassingProcess`) |
| Segmentation | `SegmentationProcess`, `core/segmentation_process.py:17` | Runs a `SegmentationMethod` per experiment on frames flagged for segmentation. | Yes |
| Pattern | `PatternProcess`, `core/pattern_process.py:24` | Collects the raw/seg frames a `PatternMethod` declared it needs, runs `generate()`, sends the result to the SLM buffer. | Yes |

`BaseProcess` (`core/base_process.py`) is a poll loop: for each registered
`(queue, handler)` it calls `queue.empty()` then `get_nowait()`, sleeps 1 ms
when no queue had work, catches and logs any handler exception (the process
keeps running), and exits when a handler returns `True` or `stop_event` is set.

### Queues (`core/queues.py`)

All queues are `multiprocessing.Queue` even though consumers are threads. This
means every item is pickled by a feeder thread, written to a pipe, and
unpickled on `get()`; the receiver gets a **copy**. Verified empirically
(`queue.Queue` returns the same object; `multiprocessing.Queue` does not; an
8 MB `uint16` frame round-trips in ~7 ms).

```
manager → microscope     manager_to_microscope     AcquisitionEventMessage, UpdatePatternEventMessage,
                                                   UpdatePositionEventMessage, "close"
manager → outbox         manager_to_outbox         "close"
manager → slm_buffer     manager_to_slm_buffer     UpdatePatternEventMessage, "close"
manager → seg            manager_to_seg            "close"
manager → pattern        manager_to_pattern        RequestPattern, "close"
microscope → manager     microscope_to_manager     UpdateZPositionMessage
(others → manager)       *_to_manager              unused
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
  process dispatches with `match msg.message:`. Most wrap an event. The
  `"close"` message is constructed ad hoc in `Manager.process`
  (`manager.py:897-900`) by setting `msg.message = "close"` on a bare `Message`.
- **Events** (`core/events.py`) describe *intent*:
  - `AcquisitionEvent` — one snap. Carries experiment name, a
    `MicroscopePosition`, `channel_id` (UUID of the `ImagingConfig`),
    scheduled wall time, `t_index`, exposure, binning, `config_groups`,
    `devices`, `needs_slm`, `sub_axes` (list used to build the HDF5 path), and
    a block of **routing/persistence booleans** (`save_output`, `save_stim`,
    `segment`, `save_seg`, `raw_goes_to_pattern`, `seg_goes_to_pattern`) plus
    method names. It is mutated by the microscope after the snap
    (`completed_time`, `pixel_width_um`).
  - `UpdatePatternEvent` — "put this experiment's current pattern on the SLM".
  - `UpdateStagePositionEvent` — "move to this position".
  - `GeneratePatternEvent` — defined but unused; `RequestPattern`
    (`pattern_process.py:219`) is what is actually sent.
- **Data** (`core/datatypes.py`) carry numpy arrays plus the originating event:
  `AcquisitionData(event, data)`, `StimulationData(event, data, dmd_pattern,
  pattern_id)`, `SegmentationData(event, data)`, `CameraPattern(experiment,
  data, slm_coords, binning)` (gets a fresh `pattern_id` UUID),
  `EventSLMPattern(event_id, pattern, pattern_unique_id)`.

---

## 3. Configuration → schedule

`directories.py:schedule_from_directory()` builds an `ExperimentSchedule`
(`core/experiments.py:254`) from an experiment directory:

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

Dry runs use `dry_schedule_from_directory()`, which additionally resolves a
TIF per position (priority: `dry_run.yml`, then position list + TIF name
matching, then TIF file names as positions) and builds a
`TimeSeriesImageSource` keyed by `(x, y)`.

---

## 4. Startup sequence (`Controller.initialize`, `controller.py:112`)

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
   model). **Segmentation is never instantiated unless a pattern asks for it.**
5. `t_gcd` of all channel `every_t` values is computed for the GUI.
6. `PatternProcess.initialize_models()` → `configure_system()` on each method
   injects camera properties and the experiment reference and applies the
   stimulation binning to `pattern_shape` / `pixel_size_um`.
7. `Manager.initialize(schedule, requirements)` precomputes, per experiment, the
   LCM of `every_t` over the required channels plus `pattern.every_t` → a
   pattern is generated only at `this_t % lcm == 0`.
8. `SLMBuffer.initialize(slm_shape, affine, names)` → blank pattern per
   experiment; `MicroscopeProcess.declare_slm()`.
9. `MicroscopeOutbox.initialize(schedule, core)` creates every HDF5 file and
   **pre-allocates every dataset for every timepoint** (see §7), then enables
   SWMR. Returns the `(file, channel)` layer list that is written to
   `all_layers.txt` (JSON) for the GUI subprocess.

`run_pyclm()` (`run_pyclm.py:66`) is the front door: finds `pyclm_config.toml`
(experiment dir, then CWD), sets up logging, builds the schedule, constructs
`Controller`, registers custom methods, hard-codes `core.setFocusDevice("ZDrive")`
(`run_pyclm.py:143`), calls `initialize`, optionally spawns the GUI as a
**separate OS process** (`subprocess.Popen` of `pyclm.gui.gui_controller`), and
calls `Controller.run()`.

---

## 5. Timing model (`Manager.process`, `manager.py:778`)

```
start = now + times.setup
for t in range(times.count):
    spin until (now - start) >= t*interval - setup        # drains *_to_manager queues, no sleep
    for i, experiment in enumerate(experiments):
        scheduled = start + t*interval + i*between
        skip if t < t_delay or (t_stop and t - t_delay >= t_stop)
        this_t = t - t_delay
        maybe send RequestPattern(t_index=this_t, ...)     # if this_t % lcm == 0
        if this_t % stim.every_t == 0:
            send UpdateStagePositionEvent (once per experiment per t)
            if stim.exposure > 0:
                send UpdatePatternEvent to slm_buffer AND microscope
                send AcquisitionEvent(needs_slm=True, t_index=t, sub_axes=[t, "stim_aq"])
        for each channel with this_t % channel.every_t == 0:
            send UpdateStagePositionEvent if not yet sent this t
            send AcquisitionEvent(t_index=t, sub_axes=[t, f"channel_{name}"])
send "close" to all five outboxes
```

Key properties:

- All events for a timepoint are emitted in one burst at the *preparatory*
  phase, `setup` seconds before the timepoint. The microscope then paces
  itself: for each `AcquisitionEvent` it sleeps until `scheduled - 0.1 s`,
  calls `waitForSystem()`, then an unconditional `sleep(1.0)`
  (`microscope.py:243`), then snaps. Channels of one experiment share one
  `scheduled` time, so they run back to back.
- `between` is a per-experiment offset. Nothing checks whether experiment *i*
  finished before experiment *i+1*'s scheduled time; overruns simply run late.
  The `todo: check if we are behind schedule` at `manager.py:791` is still open.
- The Manager never receives acknowledgements. Its only inbound message is
  `UpdateZPositionMessage` from the microscope (sent when the position mover
  reports a z change > 1 µm), which mutates `self.positions[name].z`
  (`manager.py:718-723`) so later events carry the corrected z. Inbound
  messages are only drained inside the inter-timepoint wait loop.
- `t_index` on events is the **absolute** timepoint `t`; `RequestPattern.t_index`
  is the **experiment-relative** `this_t`. See known-issues #1.

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

`MicroscopeProcess.handle_update_pattern_event` (`microscope.py:192`) does a
blocking `slm_queue.get(True, 5)` and asserts the returned `EventSLMPattern`
has the matching event id. Timeout or mismatch raises inside the microscope
loop, which has no exception guard, so the microscope thread dies and the
Controller aborts the run.

---

## 6. Data flow per frame

```
MicroscopeProcess.handle_acquisition_event
  → set config groups, device props, exposure, binning; wait; snap
  → AcquisitionData(event, img)  or  StimulationData(event, img, current_pattern, pattern_id)
  → acquisition_outbox
MicroscopeOutbox.handle_data
  → write_data(): resize pre-allocated dset, write, write_attrs, flush; dmd + pattern_id for stim;
    bump current_t_index if _timepoint_complete()
  → if event.segment:             outbox_to_seg.put(data)
  → if event.raw_goes_to_pattern: outbox_to_pattern.put(data)
SegmentationProcess.handle_segment_data
  → SegmentationData(event, model.segment(img))
  → seg_to_pattern.put(seg)           (always)
  → seg_to_outbox.put(seg)            (if event.save_seg)  → Outbox writes ".../seg"
PatternProcess
  → RequestPattern creates DataDock(time_sec, requirements) keyed "{experiment}_{t_index:05d}"
  → raw/seg arrivals fill the dock; when complete: docks.pop(), PatternContext(dock, experiment),
    model.generate(context) → CameraPattern(experiment, pattern, slm_coords, binning) → pattern_to_slm
SLMBuffer.handle_data
  → pattern_to_slm(): float[0,1] camera coords → uint8, warpAffine with affine (scaled by binning) → SLM shape
  → slm_patterns[experiment] = (pattern_id, slm_image)
SLMBuffer on UpdatePatternEvent → EventSLMPattern(event.id, slm_image, pattern_id) → slm_to_microscope
MicroscopeProcess.handle_update_pattern_event → core.setSLMImage(); remembers pattern + id for the next StimulationData
```

Who decides routing: `Manager.get_kwargs()` (`manager.py:668`) looks up the
channel's `AcquiredImageRequest` and sets `do_segmentation`, `save_segmentation`,
`raw_goes_to_pattern`, `segmentation_goes_to_pattern` on the event, but only
when a pattern is being generated this timepoint. The Outbox and Segmentation
processes just read those flags.

What a pattern method sees: `PatternContext` (`patterns/pattern.py:92`) exposes
`.time` (seconds since start, scheduled), `.raw(name)`, `.segmentation(name)`,
`.stim_raw()`, `.stim_seg()` for **the current timepoint only**. The dock is
popped and discarded after `generate()`. Any temporal state must live on the
method instance (e.g. `BounceModel.down`, `EmbryoSegmentationMethod.cached_result`).

---

## 7. HDF5 layout (one file per experiment/position)

Root attributes: `schedule_metadata` (JSON of `ExperimentSchedule.as_dict()`),
`experiment_metadata` (JSON of `Experiment.as_dict()`), `every_t` (JSON map
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
attributes pre-created by `_preallocate_attrs` (`manager.py:353`) and
overwritten by `AcquisitionEvent.write_attrs` (`events.py:160`); `dmd` also
gets `pattern_id`. Position is stored as a list of `(key, str(value))` pairs.

Why pre-allocate: SWMR readers are only guaranteed to see objects that existed
when they opened the file. (Empirically, h5py 3.14 / HDF5 1.14.6 does *not*
raise when creating attributes or datasets after `swmr_mode = True`; the
constraint is about reader visibility, not writer errors.)

Consumers of this layout: `MicroscopeOutbox._timepoint_complete`, the GUI
(`gui/gui_controller.py`), `convert_hdf5s.py`, `PatternReview`, `tests/test_dry_run.py`.

Other outputs in the experiment directory: `log.log` (file handler at INFO,
console at WARNING), `all_layers.txt` (JSON: `{"t": t_gcd, "all_layers":
["path:channel_x", ...]}`).

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
`BaseProcess` loop break at its next iteration; the outbox's `close_files()`
is **not** called on that path (data is safe because every write is followed
by `flush()`).

---

## 9. Coordinate frames and units

| Frame | Where it lives | Notes |
|---|---|---|
| Stage µm (x, y, z, extras) | `MicroscopePosition` | `PFSPositionMover` negates y when calling `setXYPosition` (`position_mover.py:82`). |
| Camera pixels, full ROI | `CameraProperties.roi` from `core.getROI()` | `(x_off, y_off, w, h)`; `get_image_shape()` returns `(h//b, w//b)`. |
| Pattern space | `PatternMethod.pattern_shape`, `pixel_size_um` | Camera ROI divided by **stimulation** binning; methods return float `[0,1]` arrays of this shape. `update_binning` uses float division so `pattern_shape` becomes floats (hence the `int(h)` casts everywhere). |
| SLM pixels | `SLMBuffer.slm_shape`, `pyclm_config.affine_transform` (2×3, camera→SLM) | `pattern_to_slm` scales the linear part by binning, converts to uint8 0–255, `cv2.warpAffine`. `PatternMethodReturnsSLM` subclasses skip the transform. |
| Time | `context.time` = scheduled seconds since `start`; `t_index` = absolute timepoint | Pattern methods convert to minutes themselves. |

The simulated core's `getROI()` returns 4× the TIF size (`simulated_core.py:192`)
so that binning-4 configs produce the TIF shape; it never actually bins images.

---

## 10. Extension points that exist today

- `PatternMethod` subclass + `Controller.register_pattern_method(name, cls)` or
  `run_pyclm(pattern_methods={...})`. Constructor gets only the TOML kwargs;
  hardware context arrives later via `configure_system`. Built-ins are
  registered in `core/patterns/__init__.py:known_models` (class-level dict
  shared by all `PatternProcess` instances).
- `SegmentationMethod` subclass (`segment(img) -> labels`), optional
  `request_resource()` for shared heavy models.
- `PositionMover` subclass (`move_to(position, core) -> (z_moved, z)`).
- `MicroscopeCoreInterface` (Protocol, `core/core_interface.py`) with
  `RealMicroscopeCore` (pymmcore-plus delegate) and `SimulatedMicroscopeCore`.
- Image sources for dry runs: `TimeSeriesImageSource` (folder/yaml, mapping,
  single stack).

## 11. Tests

`uv run --group test pytest` — 8 tests, ~60 s, all passing at HEAD (verified
2026-09-06). `test_dry_run.py` runs the whole pipeline against the simulated
core for each position-list / discovery mode and checks the HDF5 dataset
inventory, shapes and dtypes. `test_swmr.py` covers outbox init + write +
`convert_hdf5s.make_tif`. `test_base_process.py` covers the poll loop. There
are no unit tests for Manager scheduling, `DataDock`, `SLMBuffer` transforms,
pattern methods, or the shutdown protocol.
