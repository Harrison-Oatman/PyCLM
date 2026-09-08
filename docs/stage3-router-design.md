# Stage 3: router, process registration, pattern history, tracking — design

Date: 2026-09-07. Status: **implemented** on branch `Stage3-router` (from
`core-refactor` at `a8e7df7`) the same day, with all seven decisions in §8
taken as recommended. Code: `core/router.py`, `PipelineProcess` in
`core/base_process.py`, `core/writer_process.py`, `core/tracking/`,
`core/tracking_process.py`, `ExperimentState` / `PatternContext` in
`core/patterns/pattern.py`, `[tracking]` in `directories.py`, tracks in
`core/storage/ome_zarr.py` and `pyclm.io`; the default `[output] format` is
now `ome-zarr`. Tests: `tests/test_router.py`, `tests/test_tracking.py`,
extended shutdown / storage / pattern-process tests, a tracking dry run. One
refinement came out of implementation: a producer that needs every frame of
its inputs declares `continuous = True` (tracking), and the resolver honours
that regardless of the demanding consumer's cadence (§3.4).

Stage 3 closes seams **C** (routing is booleans on the event, decided by the
Manager, with shutdown counts that encode the topology), **G** (the pattern
context is a single timepoint) and the remainder of **H** (concurrency model)
from [assessment-2026-09.md](assessment-2026-09.md) §3, and ships the first
new consumer, a `TrackingProcess`, as the proof that adding a consumer is a
registration and not a surgery.

Nothing here changes the experiment TOML keys that exist today, the
`generate(context)` / `segment(img)` signatures, the plan YAML format, or the
two storage layouts. Every addition to `PatternContext` is additive.

---

## 1. Verdict and scope

Do it as planned in the roadmap, in five shippable steps (§6), with one
scope adjustment: seam H needs no code. Stage 0 already committed the code
to threads and `queue.Queue`; what remains is to make the Router the one
place a process boundary could be introduced later, and to write that down.

In scope:

- **Router.** A subscription table `(experiment, channel, kind) → consumers`
  built at `Controller.initialize` from the methods' declared requirements,
  with fan-out done by the router and fan-in for shutdown derived from the
  same table. `AcquisitionEvent` and `PlannedEvent` carry identity only.
- **Process registration.** `PipelineProcess` (consumes / produces
  declarations, one data inbox) and `Controller.add_process()`. The writer
  becomes a registered consumer like any other; `MicroscopeOutbox` stops
  being "a writer and a router in one".
- **Pattern history.** A per-experiment `ExperimentState` in the pattern
  process; `add_requirement(..., history=n)`; `context.history()`,
  `context.t`, `context.last_pattern()`.
- **Tracking.** `TrackingMethod` plug-in surface, a built-in centroid
  linker, `TrackingProcess`, `[tracking]` in the experiment TOML,
  `context.tracks()`, tracks written to OME-Zarr (`labels/tracks` and a
  `tracks.parquet` table), read by `pyclm.io`, exported to ImageJ.

Out of scope (later stages): commands/acknowledgements/hot-reload (Stage 4),
config schema and `pyclm check` (Stage 5), z / grid axes, moving
segmentation into a subprocess.

---

## 2. What exists today (verified on `a8e7df7`)

Routing is decided in three places and read in three more:

| Step | Where | What |
|---|---|---|
| decide | `core/plan.py:421` `AcquisitionPlan._routing` | From the pattern method's `AcquiredImageRequest`s: `segment`, `save_seg`, `raw_to_pattern`, `seg_to_pattern` on the `PlannedEvent`, **only when a pattern is due this timepoint** (`make_pattern`). |
| copy | `core/manager.py:456` `Manager.dispatch` | Copies the four flags plus `segmentation_method` / `pattern_method` names onto `AcquisitionEvent` (`core/events.py:58`, 20 constructor arguments). |
| act | `core/manager.py:190` `MicroscopeOutbox.handle_data` | Writes, then `if event.segment: seg_queue.put`, `if event.raw_goes_to_pattern: pattern_queue.put`. |
| act | `core/segmentation_process.py:134` `handle_segment_data` | Always forwards to pattern; to outbox `if event.save_seg`. |
| act | `core/pattern_process.py:359` `handle_from_raw` / `handle_from_seg` | Two inbound queues, one dock per `(experiment, t)`. |
| record | `core/storage/hdf5_v1.py:163` `_preallocate_attrs`, `core/events.py:143` `as_attrs` | The flags are written as dataset attributes. Nothing reads them (`grep` over `src/` and `pyclm.io`; the OME-Zarr frames table never had them). |

Shutdown counts encode the topology: `MicroscopeOutbox.handle_message`
exits at `stream_count >= 2` (microscope + segmentation),
`PatternProcess.handle_message` at `>= 2` (raw + seg), `SLMBuffer` on
`manager_done and pattern_done` (`core/manager.py:204-232`,
`core/pattern_process.py:322-330`, `core/manager.py:351`). The Manager sends
`CloseMessage` to five queues; segmentation and pattern ignore it.

Queues are fixed attributes of `AllQueues` (`core/queues.py`): 12 queues, 7
of which carry frame data along edges that would each need a twin for a new
consumer.

Adding a tracking consumer today touches (counted): `AllQueues` (+3 queues),
`AcquisitionEvent` (+2 flags), `PlannedEvent`, `_routing`, `dispatch`,
`Outbox.handle_data`, `SegmentationProcess.handle_segment_data`,
`PatternProcess` (new inbound handler, dock slot, context accessor),
`AcquiredImageRequest`, `DataDock`, both `stream_count` thresholds,
`_preallocate_attrs` / `as_attrs`, `Controller.__init__` and `initialize`,
`run_pyclm`. Fifteen sites.

Temporal state in methods lives on the instances: `BounceModel.down`
(`core/patterns/fbc_cell_movement.py:197`),
`EmbryoSegmentationMethod.cached_result`
(`core/segmentation/cellpose_segmentation.py:90`). `DataDock` is popped and
discarded after `generate()` (`core/pattern_process.py:277`).

Available for tracking without new dependencies: `scikit-image` (main
dependency; `regionprops_table`) and, through it, `scipy`
(`linear_sum_assignment`). `laptrack` 0.17 is in the `analysis` group only.

---

## 3. Design: the router

### 3.1 Vocabulary

- **Kind** of a piece of data: `"raw"` (`AcquisitionData`, including
  `StimulationData`), `"seg"` (`SegmentationData`), `"tracks"`
  (`TrackingData`, new). Each data class carries `kind` as a class attribute.
- **Key**: `(experiment, channel, kind)`. `experiment` and `channel` come
  from `data.event.index` (`{"t", "p", "c"}`, Stage 1), so identity is the
  plan's, not a flag.
- **Subscription** (frozen dataclass): `consumer`, `experiment`, `channel`,
  `kind`, `cadence` ∈ {`"always"`, `"pattern"`}, `demand: bool`.
  `"pattern"` means "only at timepoints where `plan.pattern_due(experiment,
  t)`" — today's behaviour for everything downstream of the outbox.
  `demand=False` marks a subscription that records but does not cause a
  stage to run (the writer's; see §3.4).

### 3.2 `Router` (`core/router.py`, new)

```python
class Router:
    def __init__(self, plan: AcquisitionPlan): ...
    def add(self, proc: PipelineProcess) -> None          # collects proc.subscriptions(plan), proc.produces
    def resolve(self) -> RoutingTable                       # §3.4; validates; assigns inboxes; computes fan-in
    def publish(self, data) -> int                          # any thread; puts the same object on each matching inbox
    def end_stream(self, producer: str) -> None             # producer's thread; StreamClose to a consumer when its last upstream ends
    def as_dict(self) -> dict                               # provenance (§3.7)
    undeliverable: int                                      # counter for keys with no subscriber (health, Stage 4 surfaces it)
```

`publish` is a table lookup plus `n` `Queue.put` calls; no copies (Stage 0
made queues pass references). It is called from the producer's own thread:
the microscope thread publishes raw frames, the segmentation thread
publishes labels, the tracking thread publishes tracks. **The router is not
a thread** (decision 1). Consequences worth stating:

- The writer runs concurrently with segmentation instead of ahead of it.
  Today the outbox thread compresses and writes a frame before forwarding
  it, so segmentation starts one write later than it could; after Stage 3
  the frame reaches segmentation immediately. Ordering the writer relies on
  (labels after their frame) is preserved because each consumer has one
  FIFO inbox and a producer publishes its output after it received its input.
- `end_stream` decrements a per-consumer upstream count under a lock and
  puts exactly one `StreamCloseMessage` on the consumer's inbox when the
  count reaches zero. Because every producer's data precedes its
  `end_stream` call in the same FIFO, a consumer never sees data after its
  stream close. No consumer counts anything (known-issues #17 closed).
- Errors inside `publish` are the producer's errors (logged and counted by
  its `BaseProcess` guard). Unknown keys are counted in `undeliverable`,
  logged once per key, and dropped.

### 3.3 `PipelineProcess` (`core/base_process.py`, extends `BaseProcess`)

```python
class PipelineProcess(BaseProcess):
    produces: dict[str, tuple[str, ...]] = {}       # kind produced → kinds consumed to produce it, e.g. {"seg": ("raw",)}
    def subscriptions(self, plan) -> list[Subscription]: return []   # explicit wants (pattern, writer)
    def attach(self, router, data_inbox: Queue, control_inbox: Queue | None): ...
    def handle_data(self, data) -> None: ...
    def on_stream_end(self) -> bool: ...             # default: router.end_stream(self.name); return True (exit)
```

One data inbox per process (data and `StreamCloseMessage` share it), plus
the manager's control inbox for the processes the Manager addresses
directly (microscope, SLM buffer, pattern). `DataPassingProcess` and the
`from_raw` / `from_seg` / `seg_to_outbox` pairs go away. `AllQueues` keeps
only the non-routed edges: `manager_to_microscope`, `manager_to_slm_buffer`,
`manager_to_pattern`, `microscope_to_manager`, `pattern_to_slm`,
`slm_to_microscope`. The pattern → SLM buffer → microscope path is a control
handshake with a synchronous reply (architecture-notes §5), not frame
fan-out, and stays as it is; `SLMBuffer` registers for lifecycle only.

`Manager.msgout` shrinks to the three addressed queues; `CloseMessage` goes
to those three (the outbox and segmentation never acted on it; their exit is
the stream end).

### 3.4 Resolution (`Router.resolve`)

Inputs: explicit subscriptions and `produces` maps from every registered
process, plus the experiment configs (which methods are configured).

1. **Demand.** Start from explicit `demand=True` subscriptions (the pattern
   process's, from `AcquiredImageRequest`s; a user process's). For each
   demanded `(exp, ch, kind)` with `kind != "raw"`, find the process that
   produces `kind` and add its input subscriptions `(exp, ch, kind_in)` with
   cadence `"always"` if any downstream subscription is `"always"` or the
   producer declares `continuous = True` (tracking must see every frame),
   else `"pattern"`. Iterate to a fixed point (kinds form the DAG raw → seg →
   tracks; a cycle is a configuration error).
2. **Recording.** Add the writer's `demand=False` subscriptions: `raw` for
   every planned channel, always (it also records unsaved stimulation
   events and patterns); `seg` and `tracks` for every `(exp, ch)` that step
   1 produces **and** whose method config has `save = true`. Today's rule
   "segmentation runs only if a pattern asks for it" (known-issues #23)
   becomes the general rule "a stage runs only if a demanding consumer
   needs its output; `save` decides whether the writer records it"
   (decision 2).
3. **Validate.** A demanded kind whose producer has no method configured for
   that experiment (`tracks` demanded, no `[tracking]`) raises at
   `initialize`, like an unregistered segmentation method does today. A
   configured method nobody demands logs the existing warning
   (`Controller.initialize`, extended to tracking). A `"pattern"`-cadence
   demand on a channel that the plan never acquires when the pattern is due
   raises (it would deadlock the dock; today it silently never completes).
4. **Fan-in.** `upstreams[consumer] = {producers of every kind it receives}`;
   processes with no subscriptions and nothing demanded from them are not
   started (an idle segmentation thread today costs a 1 ms poll loop; a
   never-started one costs nothing and cannot be a shutdown participant).

Worked example, the lab's usual TOML (`545` every 5, stimulation every 1,
per-cell pattern needing `seg` of `545`): pattern ← `(545, seg, pattern)`;
segmentation ← `(545, raw, pattern)`; writer ← `(545, raw)`, `(DMD, raw)`,
`(545, seg)` if `[segmentation] save`. Identical deliveries to today. Add
`[tracking] method = "centroid"` and change the pattern to
`add_requirement("545", tracks=True)`: pattern ← `(545, tracks, pattern)`;
tracking ← `(545, seg, always)`; segmentation ← `(545, raw, always)`; writer
additionally ← `(545, tracks)`. Segmentation now runs on every `545` frame
because tracking needs continuity, which is the watch-out the assessment
raised in §4.1.

### 3.5 Slimmer events

`AcquisitionEvent` loses `save_stim`, `do_segmentation`,
`segmentation_method`, `save_segmentation`, `raw_goes_to_pattern`,
`pattern_method`, `save_pattern`, `segmentation_goes_to_pattern` (8 of 20
arguments). It keeps identity (`index`, `experiment_name`, `channel_id`,
`position`), timing, exposure, binning, config groups, device properties,
`needs_slm`, `save_output`. `PlannedEvent` loses `segment`, `save_seg`,
`raw_to_pattern`, `seg_to_pattern`; `AcquisitionPlan._routing` is deleted.
`plan.requirements` and the `pattern_requires` metadata stay: they set the
pattern cadence (`_compute_lcm`) and are provenance. The `pattern_requires`
entries gain a `tracks` key (absent ⇒ false), so `PLAN_FORMAT` stays 1.
`as_attrs` / `_preallocate_attrs` drop the routing attributes; existing v1
files keep theirs, nothing reads either.

### 3.6 The writer as a consumer

`MicroscopeOutbox` becomes `WriterProcess` (`core/writer_process.py`): a
`PipelineProcess` whose `handle_data` dispatches on `data.kind` to
`FrameWriter.write_frame` / `write_labels` / `write_tracks`. `Controller.outbox`
remains as the attribute name for one release (`run_pyclm` and the tests use
`c.outbox.writer`); `MicroscopeOutbox` stays importable as an alias.
`FrameWriter.open()` gains a `routing` argument so the OME-Zarr writer
creates `labels/segmentation` and `labels/tracks` arrays exactly for the
`(experiment, channel)` pairs that will be recorded, instead of inferring
them from the pattern's requirements as it does now.

### 3.7 Provenance

`router.as_dict()` (the resolved table, per experiment: `{channel: {kind:
[consumer(cadence), ...]}}`) is logged at INFO during `initialize`, stored
under `pyclm.routing` in the OME-Zarr root attrs and as a `routing` JSON
attribute in HDF5 v1, next to the plan. An analyst can then see, per file,
why a label array is sparse (segmentation ran at pattern cadence) or dense.

### 3.8 Seam H, closed by documentation

Threads and `queue.Queue` are the model (Stage 0). The router is the only
place where a process boundary would go: a consumer whose inbox is a
`multiprocessing.Queue` and whose `publish` path pickles is a `Router`
concern, invisible to producers and to the Controller. This is noted in
architecture-notes §1; no code in Stage 3.

---

## 4. Design: pattern history (seam G)

### 4.1 Requirement API (additive)

```python
self.add_requirement("545", raw=False, seg=True, tracks=False, history=1)
self.request_stim(raw=False, seg=False, history=1)
```

`AcquiredImageRequest` gains `needs_tracks: bool = False` and `history: int
= 1`. `history=n` means "keep the last `n` deliveries of this channel to the
pattern process"; deliveries happen at the pattern's cadence (decision 3),
so a controller sampling every `lcm` timepoints sees its own last `n`
samples, which is what a PID or model-predictive method wants. A method
that needs every acquired frame of a channel is, by construction, a
tracking-style consumer and belongs in a `TrackingMethod` (§5), not in
`generate()`.

### 4.2 `ExperimentState` (`core/patterns/pattern.py`)

One per experiment inside `PatternProcess`:

- `history[(channel_id, kind)]`: `deque(maxlen=depth)` of the delivered
  data objects, depth = the largest `history` requested for that channel and
  kind (default 1, so memory is unchanged for existing methods).
- `patterns`: `deque(maxlen=pattern_history)` of `(pattern_id, camera-space
  float array)` as returned by `generate()` (default depth 2).

`DataDock` keeps its role: completeness gating for the *current* timepoint
(extended with a `tracks` slot). When a dock completes, its contents are
appended to the state's histories and the context is built on the state.
The dock is still popped; the state persists for the run.

### 4.3 `PatternContext` additions

```python
context.t                                   # plan timepoint (int); context.time stays (seconds)
context.history("545", kind="seg", n=None)  # list of arrays, oldest first, newest == current; len ≤ requested history
context.tracks("545")                       # Tracks(labels, table) for the current timepoint (§5)
context.last_pattern()                      # the previous generate() output (float array) or None
context.pattern_history(n=None)             # list of previous patterns, oldest first
```

`raw()`, `segmentation()`, `stim_raw()`, `stim_seg()`, `time` are unchanged.
`ZooContext` (`core/patterns/zoo.py`) implements the same methods (history
returns `[image]`, `tracks` a trivially-labelled table, `last_pattern`
`None`) so the docs zoo keeps rendering every method.

---

## 5. Design: tracking, the first plug-in

### 5.1 Plug-in surface (`core/tracking/`, new)

```python
class TrackingMethod:
    name = "base"
    def __init__(self, experiment_name: str, channel: str, **kwargs): ...
    def track(self, labels: np.ndarray, t: int, pixel_size_um: float) -> tuple[np.ndarray, list[TrackRow]]:
        """Relabel `labels` with stable track ids; return the rows for this timepoint."""
```

`TrackRow` (NamedTuple): `track_id, label, y, x, area, parent` (pixels of
the binned frame; µm are derived on export from `pixel_size_um`). The
interface is "labelled regions in, relabelled regions plus a table out", so
it serves cells, organoids and tissue regions alike; the method decides
what a region is. One method instance per `(experiment, channel)`; the
instance holds whatever state it needs (the last centroids, its id
counter), the same way segmentation methods do.

Built-in (decision 5): **`centroid`** — `regionprops_table` for centroids
and areas, Hungarian assignment on centroid distance
(`scipy.optimize.linear_sum_assignment`, already a transitive dependency)
gated by `max_distance_um` (default 20), unmatched regions get new ids,
`parent = 0`. No divisions, no gap closing; it is the honest baseline and
it is fast (< 5 ms for a few hundred regions). A **`laptrack`** adapter
that re-links a sliding window of the last `window` timepoints with
LapTrack (gap closing and splitting, `parent` filled) is registered only
if `laptrack` imports; it is a follow-up unless decision 5 says otherwise.

Registration mirrors the others: `Controller.register_tracking_method(name,
cls)`, `run_pyclm(tracking_methods={...})`.

### 5.2 `TrackingProcess` (`core/tracking_process.py`)

`produces = {"tracks": ("seg",)}`. On `SegmentationData` for `(exp, ch)`:
`labels, rows = method.track(...)`; publishes `TrackingData(event, labels,
rows)` (`kind = "tracks"`, sharing the acquisition event like
`SegmentationData` does). Constructed by the Controller for every
experiment with a `[tracking]` section, started only if something demands
`tracks` (§3.4).

### 5.3 Configuration

```toml
[tracking]
method = "centroid"     # or a registered name
max_distance_um = 20    # remaining keys are method kwargs, as for [segmentation]
save = true             # record tracks in the output (default true)
```

`directories.experiment_from_toml` parses `[tracking]` exactly like
`[segmentation]` (a `MethodBasedConfig`; absent ⇒ `"none"`);
`Experiment.tracking` is added and included in `as_dict()` so it lands in
the outputs' `experiment_metadata`.

### 5.4 Storage, reading, export

- **OME-Zarr** (`OMEZarrWriter.write_tracks`): relabelled masks as an NGFF
  label image `imaging/labels/tracks/0` `(T, C, Y, X) uint16` on the
  channel's cadence group (created at `open` from the routing table); rows
  appended to `tracks.parquet` in the experiment directory (columns:
  `experiment, t, channel, track_id, label, y, x, y_um, x_um, area, parent`),
  rewritten as the run proceeds like `frames.parquet`, with `tracks.csv` at
  close.
- **HDF5 v1**: `FrameWriter.write_tracks` has a default implementation that
  logs once and drops; v1 keeps writing frames and labels as today and does
  not grow a new dataset kind (decision 6).
- **`pyclm.io`**: `GroupData.tracks(i, channel)` (labels array or `None`),
  `GroupData.has_tracks`, `ExperimentData.tracks` (pyarrow table for this
  experiment). `export_imagej` adds the tracked labels after the
  segmentation labels (LUT distinct from the yellow seg LUT), so a Fiji
  user sees ids that persist across frames.
- `data_format.md` gains the `labels/tracks` line and the tracks table.

---

## 6. Staged path (each step shippable, all tests green)

| Step | What | Size |
|---|---|---|
| **3a** | `Router`, `PipelineProcess`, `WriterProcess`; events slimmed; `AllQueues` trimmed; shutdown derived; `Controller` builds and resolves the router. Behaviour identical to today; existing dry runs and the storage tests are the regression. | 3 days |
| **3b** | `Controller.add_process()`, routing provenance in both formats, `register_tracking_method` scaffolding, `[tracking]` parsing, `Experiment.tracking`. | 1 day |
| **3c** | `ExperimentState`; `history`, `t`, `last_pattern`, `pattern_history`; `ZooContext` parity; `custom_pattern_methods.md` updated. | 2 days |
| **3d** | `TrackingMethod`, `centroid`, `TrackingProcess`, `TrackingData`, `write_tracks` in OME-Zarr, `tracks.parquet`, `pyclm.io` and export support, `context.tracks()`, a tracking dry run, a `tracking.md` user page. | 4 days |
| **3e** | Flip `[output] format` default to `ome-zarr` (decision 6); docs and config samples updated; v1 writer kept. | ½ day |

About two weeks, matching the roadmap's estimate. 3a is the risky step
(it touches every process); it is done first so the rest lands on a
verified base.

---

## 7. Tests and documentation

New tests:

- `test_router.py`: resolution for the three configurations in §3.4
  (pattern-only, pattern + tracking, nothing demanded); cadence filtering
  against a plan over `every_t` / `pattern_every_t` combinations; the same
  object delivered to every subscriber; unknown key counted and dropped;
  stream close delivered exactly once after the last upstream, under
  concurrent producers; validation errors (missing producer, undeliverable
  pattern-cadence demand).
- `test_pattern_process.py` (extended): history depth and order,
  `last_pattern`, `tracks` slot gating, `context.t`.
- `test_tracking.py`: the centroid linker on synthetic moving blobs (ids
  stable under motion below `max_distance_um`, new ids beyond it,
  disappearance); `TrackingProcess` relabels and publishes rows.
- `test_shutdown.py` (extended): graceful drain with tracking registered and
  with a user process registered; forced stop unchanged.
- `test_dry_run.py`: a tracking configuration (`[tracking] method =
  "centroid"`, a stub pattern using `context.tracks`) on OME-Zarr:
  `labels/tracks` present, `tracks.parquet` rows, export contains the
  tracked-labels channel.
- Existing tests updated for the slimmer events and the trimmed
  `AllQueues` (`test_manager_scheduling.py::test_close_is_sent_to_every_process`,
  `test_plan.py` routing-flag assertions become router tests).

Documentation: architecture-notes §1, §2, §6, §8, §10, §11 rewritten for
the router; known-issues #17 and #18 closed, #23 restated as the rule;
`custom_pattern_methods.md` (history / tracks / last_pattern API);
`experiment_tomls.md` (`[tracking]`); new `tracking.md`; `data_format.md`
(tracks layout, table, export); `api/` autodoc for `pyclm.core.tracking`.

---

## 8. Decisions (all approved as recommended, 2026-09-07)

1. **Router placement.** *Recommended:* the router is an object called from
   the producer's thread (§3.2); the outbox thread becomes a pure writer.
   *Alternative:* keep a router thread in the outbox's place (one extra hop
   per frame, and the writer stays ahead of segmentation as today).
2. **Activation rule.** *Recommended:* a processing stage (segmentation,
   tracking) runs for a channel only when a demanding consumer (a pattern
   method or another stage) needs its output; `save = true` only decides
   whether the writer records it (today's #23 behaviour, generalised).
   *Alternative:* a configured `[segmentation]` / `[tracking]` with `save =
   true` runs on its own, so labels or tracks can be produced for later
   analysis without a pattern that uses them. The alternative would make
   any TOML with an unused `[segmentation]` section start running Cellpose
   on every pattern-due frame.
3. **History semantics.** *Recommended:* `history=n` keeps the last `n`
   deliveries at the pattern's cadence (§4.1); every-frame consumers are
   tracking methods. *Alternative:* `history` forces `"always"` cadence for
   that channel, so segmentation runs on every frame whenever a method asks
   for history.
4. **Tracks in the context.** *Recommended:* `context.tracks(channel)`
   returns relabelled labels plus the row table, and `context.segmentation`
   keeps returning the raw segmentation. *Alternative:* when tracking is
   configured, `segmentation()` transparently returns the tracked labels.
5. **Built-in tracker.** *Recommended:* the `centroid` linker (no new
   dependency) ships in Stage 3; the LapTrack adapter follows when a
   tracking experiment needs divisions or gap closing. *Alternative:* ship
   the LapTrack adapter now as well, which moves `laptrack` into the main
   dependencies.
6. **Tracks on disk and the default format.** *Recommended:* tracks are
   written only by the OME-Zarr writer (HDF5 v1 logs once and skips them),
   and, since Stage 2 has now run on the microscope, this stage flips the
   default `[output] format` to `ome-zarr` (step 2d; `hdf5` stays available
   by setting it). *Alternative:* keep `hdf5` as the default and add a
   `tracks` dataset to the v1 layout.
7. **Public registration API.** *Recommended:* document
   `Controller.add_process()` and `PipelineProcess` as an advanced extension
   point alongside `register_tracking_method`. *Alternative:* keep
   `add_process` internal for now and expose only the method registries.

---

## 9. Additions after implementation (2026-09-08)

Two extensions came out of writing the tracking examples, both approved
and implemented the same day.

**Named segmentations.** An experiment can configure several segmentations
of the same frames: `[segmentation]` is the default table and
`[segmentation.<name>]` sub-tables are named ones, each with its own method
and kwargs (`Experiment.segmentations`; `experiment.segmentation` stays the
default entry). A pattern method asks with `add_requirement(channel,
seg="nuclei")` (or a list); `AcquiredImageRequest.kinds` lists the routing
kinds. The router vocabulary (`core/kinds.py`) grew one rule: a named
segmentation travels as kind `seg:<name>`, and producers are registered by
the base kind, so one `SegmentationProcess` serves every name. It runs each
demanded name of a channel at that name's own cadence
(`Router.produced_by`, `SegmentationProcess.segmentations_for`), so a
segmentation only a pattern needs is not computed on the frames only a
tracker sees. `[tracking] segmentation = "<name>"` picks which table the
tracker links (`TrackingProcess.inputs`, the `inputs` hook the router now
consults before `produces`). OME-Zarr stores one label image per table
(`labels/<name>`, listed in the NGFF `labels` attribute); HDF5 format 1
keeps the default one and warns once about the rest; `pyclm.io` exposes
`label_names` and `labels(i, channel, name)`; the export adds one label set
per table. This closes the "one segmentation configuration per experiment"
limit that made a KTR readout without a nuclear marker impossible.

**Measurement toolbox** (`core/measure.py`, exported from `pyclm`):
`Regions` (a label image with `ids`, `measure(image, stat)`, `paint(values)`,
`areas`, `centroids`, `select`, `owner_of`), `PerTrack` (a value per track
id with `update` / `get(ids, default="median")`), and
`nuclear_cytosolic_ratio(nuclei, cells, image)`. `Tracks` is a `Regions`;
`context.regions(channel, name)` wraps a segmentation. The documented
examples (`documentation/examples/`) were rewritten on it; the KTR clamp is
the third.

Tests: `tests/test_named_segmentation.py`, `tests/test_measure.py`, named
labels in `tests/test_storage.py`, a named-segmentation dry run, and the
example tests.
