# PyCLM core architecture assessment — September 2026

> **Status.** Plan approved. Stage 0 (hygiene) implemented on branch
> `Stage0-hygiene` on 2026-09-06; see [known-issues.md](known-issues.md) for
> what each fix covered and what remains. Stages 1–5 are pending; the
> useq-schema spike is the next step.

Scope: the core modules (Manager, MicroscopeProcess, MicroscopeOutbox,
SLMBuffer, SegmentationProcess, PatternProcess) and the code that feeds them
(`controller.py`, `directories.py`, `core/events.py`, `core/datatypes.py`,
`core/patterns/pattern.py`), read on branch `core-refactor` at `7af035a`. The
factual map of the runtime is in [architecture-notes.md](architecture-notes.md);
the itemised bug list is in [known-issues.md](known-issues.md). This document
is the opinion: what holds up, what does not, and what to do about it.

---

## Summary

**The architecture's shape is right and should be kept. A rewrite is not
warranted. A staged refactor of the core's internal seams is.**

The pipeline decomposition (hardware / storage / segmentation / pattern /
SLM as independent workers fed by a scheduler), the plugin surfaces
(`PatternMethod`, `SegmentationMethod`, `PositionMover`, `MicroscopeCoreInterface`),
the TOML-driven configuration, the one-interval-delayed closed loop with
`pattern_id` provenance, the virtual microscope, and the end-to-end dry-run
tests are all sound decisions that a redesign would reproduce.

What does not hold up is that five cross-cutting concerns are each implemented
*implicitly*, in several places, as a by-product of the original two-axis
(time × channel), one-position-per-experiment, fixed-schedule design. Every
one of the five planned features collides with two to four of those same
seams, so fixing the seams once is cheaper than working around them five
times.

| Seam | Where it lives today | Tracking | Runtime edits | Z-stacks | Grids | Interactive setup |
|---|---|---|---|---|---|---|
| **A.** Schedule is procedural and duplicated ×4 | `Manager.process`, `Outbox.initialize`, `Outbox._timepoint_complete`, GUI | | ● | ● | ● | ● |
| **B.** Storage schema fixed at start, per (t, channel) 2-D dataset | `Outbox.initialize` + SWMR | ● | ● | ● | ● | |
| **C.** Routing encoded as booleans on the event, decided by the Manager | `AcquisitionEvent`, `Manager.get_kwargs`, Outbox/Seg branches, shutdown counts | ● | | | | |
| **D.** Experiment ≡ position ≡ HDF5 file | `ExperimentSchedule`, `Manager.positions`, `SLMBuffer.slm_patterns`, dock keys | | | | ● | |
| **E.** No control plane, no feedback from the microscope, GUI out-of-process and read-only | `Manager` inbox, `run_pyclm` subprocess GUI | | ● | | | ● |
| **F.** Config has no schema | `directories.py` hand parsing | | ● | ● | ● | ● |
| **G.** Pattern context is single-timepoint; docks are discarded | `DataDock`, `PatternContext` | ● | | | | |

Plus one housekeeping seam, **H**: the concurrency model hedges between
threads and processes (`multiprocessing.Queue` under a `ThreadPoolExecutor`),
and the Manager's wait loop spins. Neither blocks a feature; both should be
settled early because they shape everything else.

Verified facts this assessment rests on: all 8 tests pass at HEAD (~60 s); the
`t_delay` indexing bug is real and silently misattributes frames
(known-issues #1); queues copy every frame even between threads (~7 ms per
8 MB frame, not a bottleneck); HDF5 1.14 accepts attribute and dataset
creation after SWMR mode is enabled, so the fixed schema is a reader-visibility
choice, not a library limit.

Rough overall size of the recommended refactor: 6–9 weeks of focused work
spread over stages that each leave the code shippable and the dry-run tests
green. About half of `core/manager.py` and the event/flag plumbing changes;
`microscope.py`, `position_mover.py`, the core interfaces, the method
implementations, and most of `directories.py` survive with small edits. The
public surfaces (TOML format, `generate(context)`, `segment(img)`, the HDF5
readers) can be preserved or versioned rather than broken.

---

## 1. What works and should be kept

- **Worker decomposition and back-pressure by design.** The microscope thread
  only ever waits on its own inbox and on the SLM handshake; slow segmentation
  or pattern generation never delays acquisition, it just yields a stale
  pattern. That is the correct failure mode for a live experiment, and the
  `pattern_id` written next to every stimulation frame makes it auditable.
  This property must survive the refactor.
- **Declarative requirements.** `add_requirement()` / `request_stim()` →
  `AcquiredImageRequest` → `DataDock` is a clean way for a method to say what
  it needs, and it already drives whether segmentation runs at all. Extending
  it (e.g. `needs_tracks`, `history=3`, `z="max"`) is the natural path for
  tracking and z-stacks.
- **Hardware behind a Protocol, with a simulator.** `MicroscopeCoreInterface`
  + `SimulatedMicroscopeCore` + `TimeSeriesImageSource` make the whole
  pipeline testable without hardware; the integration tests prove it. Any
  refactor should be judged by "does the dry run still pass".
- **Position movers.** Isolating PFS-specific behaviour into `PositionMover`
  is the right cut for hardware autofocus and will be the right place for
  z-stack stepping policy.
- **Provenance in the file.** Schedule and experiment metadata as JSON attrs,
  per-frame attrs, `current_t_index` for live readers: the intent is right
  even though the mechanism is rigid.
- **Graceful drain on completion.** The stream-close cascade is fragile to
  extend (seam C) but correct for the current topology.
- **Zoo / docs generation from method metadata.** `ZooMeta` and
  `make_zoo_subclass` are exactly the machinery a "preview this pattern on a
  snapped image" feature needs.

---

## 2. How data moves and where it is modified

Tracing one frame end to end (details in architecture-notes §5–§6):

1. **Manager** decides everything about the frame up front: position,
   channel config, exposure, binning, where it will be stored (`sub_axes`),
   and who will consume it (`segment`, `raw_goes_to_pattern`,
   `seg_goes_to_pattern`, `save_*`). It emits the event and forgets it. It
   never learns whether or when the frame was taken.
2. **Microscope** mutates the event in place (`completed_time`,
   `pixel_width_um`), wraps it with the image, and for stimulation frames
   attaches the SLM image and `pattern_id` it applied. It is the only module
   that talks to hardware and the only one that blocks synchronously on
   another module (the SLM buffer).
3. **Outbox** is two modules in one: a writer and a router. It writes,
   flushes, updates `current_t_index`, and then forwards the *same* object to
   segmentation and/or pattern based on the event's flags. The HDF5 write is
   the only place per-frame metadata is persisted.
4. **Segmentation** produces a new `SegmentationData` sharing the original
   event and forwards to pattern unconditionally, to outbox if `save_seg`.
5. **Pattern** collects frames into a per-(experiment, timepoint) dock,
   runs `generate()` once the dock is complete, discards the dock, and emits a
   float camera-space pattern.
6. **SLM buffer** converts to SLM space (affine, binning scale, uint8) and
   *overwrites* the experiment's current pattern; the microscope pulls it on
   the next stimulation event.

Two observations follow. First, the data objects are immutable in practice
except for the event's completion fields, which is good; but because
`multiprocessing.Queue` copies on every hop, "the same event object" is
actually several independent copies, and any future in-place update (e.g. the Manager
correcting a position) does not propagate to events already in flight, which
is fine today and must stay explicit. Second, the only place per-frame state
persists is the HDF5 file; there is no in-memory record of what has been
acquired. That is why the GUI reads the file, why the Manager cannot report
"behind schedule", and why a tracking module has nowhere to look for history.

---

## 3. The seams

### A. The schedule is a procedure, not a data structure

`Manager.process` *is* the schedule: the loop over `t`, the `t_delay`/`t_stop`
gating, the `every_t` modulus, the stim-before-channels ordering, the
per-experiment time offsets. The Outbox needs the same information to
pre-allocate datasets and to decide when a timepoint is complete; the GUI
needs it to know which frames exist. All four re-derive it, and known-issues
#1 shows they have already diverged once (absolute vs relative `t`).

Consequences: z-stacks and grids are new axes in this loop, and every
consumer must learn them; runtime edits are edits to a loop that is running;
timing feasibility ("will 3 channels × 5 z × 4 tiles fit in `interval`?")
cannot be computed because nothing enumerates the events before they happen.

Direction: materialise the plan. An `AcquisitionPlan` object built from
`ExperimentSchedule` that can answer `events_at(t)`, `is_scheduled(exp,
channel, t)`, `expected_datasets()`, and `estimated_duration(t)`. Give each
planned acquisition a stable identity (an `index` dict such as
`{"t": 12, "p": "bar10.00", "c": "545", "z": 0, "g": 0}` replacing the
`sub_axes` list) that the Manager stamps on the event, the Outbox uses to
derive the storage path, the Pattern process uses to key docks, and the GUI
uses to place frames. The Manager becomes a thin executor of the plan.

**Strong recommendation: spike whether `useq-schema` can be the plan
representation before writing a new one.** pymmcore-plus already depends on
it (it is in `uv.lock`), so there is no new dependency. `MDASequence` models
time, position, channel, z and grid plans, allows per-position sub-sequences
(which is how "different `every_t` per experiment" would be expressed), and
`MDAEvent` carries an `index` dict, positions, exposure, channel config, device
properties, and free-form metadata, and serialises to JSON/YAML for
provenance. pymmcore-widgets provides Qt editors for positions, z-plans,
grid-plans and channels on top of it, which bears directly on interactive
setup. PyCLM's per-timepoint "prepare-then-pace" timing and its SLM handshake
are custom, so the recommendation is to adopt the *event model* and keep the
custom executor, not to adopt pymmcore-plus's MDA runner wholesale. The spike
should try to express the two test experiments plus one z-stack experiment as
an `MDASequence` and iterate its events in a dry run; if it cannot express the
multi-experiment schedule cleanly, write the small in-house plan instead, but
keep `index` dict semantics compatible.

Checked while writing this (useq-schema 0.8.1 from the project venv): a
sequence with a 60 s × 3 time plan and two positions, the second carrying its
own sub-sequence with a different channel list and a z-plan, enumerates 15
events with `index` dicts like `{t: 0, p: 1, c: 0, z: 2}`, per-event
`z_pos`, `min_start_time`, `properties`, `metadata`, and an `slm_image`
field, and the whole sequence round-trips through JSON. What the spike still
has to establish is per-channel `every_t` within one position (useq's time
plans are per sequence, so this may need one sub-sequence per cadence or a
post-filter), `t_delay`/`t_stop`, and the stim-before-channels ordering.

### B. The storage schema is fixed at start

Because SWMR readers only see objects that existed when they opened the file,
`MicroscopeOutbox.initialize` pre-creates one 2-D dataset per (timepoint,
channel, kind). Everything that changes the set of frames at runtime, or adds
an axis, or wants variable-length output (a track table) conflicts with this,
and the failure mode is silent frame loss (known-issues #6). The GUI's
Windows-specific reopen-per-refresh and the `PermissionError` retry loops are
further costs of the current file-level design.

Direction (within HDF5): pre-allocate **per channel**, not per timepoint:
`channel_545/data` shaped `(0, H, W)` with `maxshape (None, H, W)` (and
`(None, Z, H, W)` for stacks), appended on write, plus one growable table
dataset `frames` with columns (index fields, wall time, scheduled time,
exposure, position, pattern_id, config summary) replacing per-dataset
attributes. This is the canonical SWMR use case (append to an existing
dataset), keeps the live GUI working with simpler code (dataset index =
acquisition order), keeps `t_count` from being a hard cap, and gives tracking
a natural home (`tracks` table, growable). Version the layout with a root
attribute and keep `convert_hdf5s` and the GUI reading both versions for a
while.

Alternative worth a look when z-stacks or grids actually land: OME-Zarr.
Chunk-per-frame directory stores need no pre-allocation, have no
writer/reader locking, are the native (T, C, Z, Y, X) layout, and napari reads
them directly. The cost is migrating the converter, notebooks and GUI. Do not
do this as part of the first stages; do introduce a `Writer` interface (seam
C) so the choice can be made later without touching the pipeline.

### C. Routing is encoded on the event, decided by the Manager

Who consumes a frame is a set of booleans on `AcquisitionEvent`, computed by
`Manager.get_kwargs` from the pattern's requirements, interpreted by the
Outbox's `handle_data` and the Segmentation process, and mirrored into the
HDF5 attribute schema. The shutdown cascade hard-codes how many upstreams
each process has. Adding one consumer (tracking) touches: `AllQueues`, the
event class, `get_kwargs`, `Outbox.handle_data`, `SegmentationProcess`,
`PatternProcess` (a new inbound handler, dock slot, context accessor),
`AcquiredImageRequest`, both `stream_count` thresholds, `_preallocate_attrs`,
`write_attrs`, and the Controller wiring. That is the honest answer to "does
the structure allow a tracking module": yes, at roughly a dozen touch points,
each easy to forget.

Direction: split the Outbox into a `Writer` and a `Router`. The Router owns a
subscription table built at initialisation from the methods' requirements
(`{(experiment, channel_id, kind): [consumers]}`) and fans out; the event
carries identity only. Processes register with the Controller
(`add_process(proc, consumes=[...], produces=[...])`), which builds the
queues and derives the shutdown fan-in counts from the same table. Then a
`TrackingProcess` is a registration, not a surgery.

### D. Experiment ≡ position ≡ file

`ExperimentSchedule` asserts the experiment and position key sets are equal;
the Manager, SLM buffer, pattern docks, outbox files, and the HDF5 naming all
key on the position label. This is exactly right for "many independent
experiments, one FOV each" and exactly wrong for grid acquisition, where one
experiment instance spans several FOVs and one pattern may span the grid.

Direction: separate *experiment config* (a TOML) from *experiment instance*
(config + one or more positions + one output) and give positions an
identity within the instance (the `p`/`g` index). Most code only needs to
key on the instance; the SLM buffer and the pattern process need
per-FOV entries; the writer needs a tile axis or stitching. MicroManager's
grid-creation naming (`Grid-1-Pos_000_000`) suggests a label convention
`exp.grid1.000_000` can carry this through the existing position-list route
without new files.

### E. No control plane

The Manager has one inbound message (`UpdateZPositionMessage`), drains it only
while idle, never acknowledges anything, and never learns whether the
microscope is on time. The GUI is a separate OS process that reads the HDF5
files; the only channel between them is `all_layers.txt`. So "update a
position while running" has a precedent in the code (the z-correction) but no
transport a user can reach.

Direction: a small typed command set into the Manager (`SetPosition`,
`SetChannel`, `SetPatternParams`, `Pause`, `Resume`, `StopExperiment`,
`ExtendSteps`) applied at a defined point (between timepoints), each
acknowledged and **recorded in the output file** (an `events` table) so
analysis can see what changed when. For transport, the cheapest thing that
matches the lab's TOML-first habits is **hot-reload of the experiment
directory**: watch `PositionList.pos` and the experiment TOMLs, diff against
the running config, and apply the safe subset (positions, exposures, config
groups, pattern kwargs, `t_stop`) at the next timepoint. That gives runtime
edits with the tools users already have. A socket protocol for the GUI can
come later on the same command set. In parallel, add microscope → manager
acknowledgements (`EventDone(index, completed_time)`) so the Manager can keep
an in-memory acquisition record, report lateness, and eventually skip or
reschedule.

### F. Configuration has no schema

`directories.py` parses TOML by hand with `.get` defaults and `pop`s, unknown
keys become method kwargs, and errors surface as `KeyError`/`AssertionError`
during initialisation. `documentation/experiment_tomls.md` is still
"(Content to be added)". Interactive setup, validation, hot-reload diffing,
and documentation all need the same thing: a declared model.

Direction: pydantic (or dataclasses + a validator) models for
`pyclm_config.toml`, `schedule.toml`, and the experiment TOML, with a
`pyclm check <dir>` command that parses, cross-checks positions ↔ TOMLs,
verifies config groups/presets exist in the loaded MicroManager config,
checks method kwargs against the method's `__init__` signature, and prints a
timing budget from the plan (seam A). Add a `format_version` key. This is
independent of the core refactor and is the highest-value, lowest-risk item
for day-to-day usability.

### G. The pattern context is a single timepoint

`DataDock` holds exactly the raw/seg frames one `generate()` call asked for
and is discarded afterwards. Any method that needs the past keeps its own
state (`BounceModel.down`, `EmbryoSegmentationMethod.cached_result`). A
tracking module is *about* the past; so is any real controller (PID,
model-predictive, trajectory following).

Direction: a per-experiment `ExperimentState` in the pattern process (bounded
history of raw/seg/tracks per channel, last N patterns, the applied
`pattern_id`s) that `PatternContext` exposes additively:
`context.history("545", seg=True, n=3)`, `context.tracks()`,
`context.last_pattern()`. The requirement API declares the depth so the
router knows what to retain. `generate(context)` stays the only method a user
implements.

### H. Concurrency model

Threads under `ThreadPoolExecutor`, but `multiprocessing.Queue`, `stop_event`
as a `threading.Event`, the live core object handed to the microscope thread,
and class-level registries: this only works as threads, so commit to threads.
The heavy stages (Cellpose on GPU, numpy/OpenCV, h5py, pymmcore) release the
GIL, so throughput is not the concern; the Manager's spin loop is (known-issues
#2). Switch to `queue.Queue` (zero-copy, reliable `empty()`, `task_done/join`
for tests), add the sleep, and keep "segmentation in a subprocess" as a
possible later optimisation behind the Router.

---

## 4. Feature-by-feature

### 4.1 Tracking module (cells / organoids / tissues across timepoints)

**Fits today?** Partially. There are two viable placements:

- *Minimal, no core change:* a `SegmentationMethod` subclass that keeps
  history and returns masks relabelled by track id (e.g. wrapping LapTrack,
  which is already a dependency of the analysis group). Pattern methods see
  consistent labels via `context.segmentation()`. Limits: no track table on
  disk, no GUI overlay, no lineage, state hidden inside the seg process. This
  is a fine way to get a tracking-driven experiment running next month.
- *First-class:* a `TrackingProcess` consuming segmentation output,
  producing relabelled masks plus a per-timepoint table (track id, t,
  centroid, area, features, parent), stored as a growable table (seam B) and
  exposed through `context.tracks()` (seam G). Requires seams B, C, G.

Per-organoid/tissue tracking is the same shape with region-level features
instead of cell-level; the interface should be "labelled regions + table",
not "cells".

**Watch-outs:** tracking wants every timepoint of its channel even when the
pattern only regenerates every `lcm` timepoints; today frames are routed to
consumers only when a pattern is being generated (`get_kwargs` with
`make_pattern=False` sends nothing downstream). The Router must distinguish
"tracking subscribes always" from "pattern subscribes at its cadence".

### 4.2 Update positions / settings while the experiment is running

**Fits today?** Mechanically yes for the small stuff, with no user-facing
transport. The Manager reads `self.positions[name]` and each `ImagingConfig`
afresh every timepoint, so a message that mutates them takes effect at the
next timepoint and is recorded in the per-frame attrs. The z-correction path
is the template. Pattern parameters need a message to the pattern process and
an `update_params()` hook on `PatternMethod`.

**Does not fit today:** anything that changes the *set* of frames (every_t,
channels, extending steps, adding a position) because of seam B; pause/resume
is possible by shifting `start_time` but timestamps need care; and there is no
transport (seam E). Recommended path: command set + hot-reload of the
experiment directory + an `events` table in the output, then the GUI.

### 4.3 Z-stacks

**Fits today?** The hooks exist (`sub_axes` was designed for extra axes;
`core.setPosition(z)` is on the interface) but every consumer assumes two
levels and 2-D arrays: the Outbox pre-allocation and completion check, dock
slots, `PatternContext`, the GUI, `convert_hdf5s`, and the simulated source.
Timing is the other blocker: N slices × (1 s fixed settle + exposure) per
channel per position, with no way to know in advance whether it fits the
interval.

Design decisions to make explicitly: per-slice events (simplest; the
executor just gets more events) versus one event executed as a burst
(faster, keeps pymmcore hardware sequencing open); how PFS interacts with
z-stepping (`PositionMover` policy: step the PFS offset, or unlock/relock);
what a pattern method receives (a stack, or a projection declared in the
requirement, e.g. `add_requirement("545", seg=True, z="max")`); whether
segmentation runs per slice or on the projection. Storage is `(T, Z, H, W)` per
channel under seam B. Requires seams A, B, F; benefits from useq's `z_plan`.

### 4.4 Grid-based acquisition

**Fits today?** No, if it means tiles of one sample under one experiment; yes,
if it means a grid of independent positions (that is just more positions in
the list). The tiled case breaks seam D everywhere the position label is the
key. Beyond the data model, the two design questions are pattern scope
(per-tile patterns are cheap: the method sees one FOV as today; grid-wide
patterns need a stage-µm frame, stitching of segmentations, and cutting the
global pattern into per-tile SLM images through the per-FOV affine) and
timing (stage travel across tiles eats the `between` budget). Requires seams
A, D, B; useq's `grid_plan` plus MicroManager's grid position naming cover the
plan side.

### 4.5 More user-friendly experiment setup / interactivity

**Fits today?** Nothing blocks it, but nothing supports it either. The
pieces that exist: the dry run, the zoo renderer (render any method on a
sample image), the position-list importers, the live GUI, and the fact that
`Controller.__init__` already owns a live core before the run starts (so
snap/move/preview before `run()` is possible).

Ordered by value for effort:

1. `pyclm check <dir>` validation with a timing budget (seam F + A). No UI,
   catches most mistakes before the microscope is touched.
2. Pattern preview: "snap this position, run method X with these kwargs,
   show the pattern in camera and SLM space" reusing `make_zoo_subclass`.
3. TOML authoring from the schema (a form per section), and a positions
   editor that snaps and names positions with the experiment linkage built
   in. If the useq spike succeeds, pymmcore-widgets gives most of this.
4. Run control (start/pause/abort) and runtime edits in the GUI over the
   command set (seam E). This is the point at which running the GUI
   in-process (napari on the main thread, Controller in a worker thread)
   becomes simpler than the subprocess design.

---

## 5. Is a major refactor needed, and what would it entail

**Needed: yes, for the internals. Rewrite: no.** The recommended order keeps
the dry-run tests green at every stage and lets any stage be shipped alone.
Sizes are rough and assume one person with the codebase already in their head.

| Stage | Goal | Main changes | Tests to add | Size |
|---|---|---|---|---|
| **0. Hygiene** | Remove the hazards that would confuse everything after | `queue.Queue`; sleep in Manager wait loop; fix #1 (`t` indexing); close files on force-stop; per-run logging; exception guard + error counter in the microscope loop; configurable settle time and focus device; delete dead code; `print` → `logger` | Unit test for dock keying with `t_delay`; shutdown-protocol test | 1–2 days |
| **1. Plan** | Seam A. Schedule becomes data with an `index` per acquisition | `AcquisitionPlan` (or `useq.MDASequence` after a spike); Manager walks it; Outbox and GUI derive from it; `sub_axes` → `index`; one `is_scheduled` | Plan enumeration tests (every_t, t_delay, t_stop, lcm); timing-budget test | 1–2 weeks incl. spike |
| **2. Storage v2** | Seam B. Growable per-channel datasets + `frames` table; format version | Writer rewrite; GUI and `convert_hdf5s` read v1 and v2; a `pyclm.io` reader module replacing the duplicated helpers | Write/read round-trip; SWMR live-read test; v1 compatibility test | 1–2 weeks |
| **3. Router** | Seams C, G, H. Writer/Router split; subscriptions; process registration; `ExperimentState` history in the pattern process | Outbox split; `Controller.add_process`; derived shutdown fan-in; `PatternContext.history/tracks` (additive); `TrackingProcess` as the first plug-in | Router fan-out tests; tracking plug-in dry run | 1–2 weeks |
| **4. Control plane** | Seam E. Commands, acknowledgements, hot-reload, events table | Manager command inbox; `EventDone` from microscope; directory watcher + config diff; `PatternMethod.update_params`; `events` table in the file | Command application at timepoint boundary; provenance in file | 1 week (+GUI transport later) |
| **5. Schema & setup** | Seam F, then interactivity | pydantic models + `pyclm check`; generated TOML docs; pattern preview; positions editor / pymmcore-widgets if useq was adopted | Validation tests | 1–2 weeks |
| **Z / Grid** | Features on top | Per the decisions in §4.3–4.4 | Dry runs with a z-stack TIF source and a tiled source | after 1–3 |

Stage 5's first half (schema + `check`) is independent of the rest and can be
done first if usability is the pressing need.

What to hold stable through all of this, because the lab's experiments, the
docs, and the zoo depend on them: the experiment TOML keys (add, don't
rename; add `format_version`), `PatternMethod.generate(context)` and the
requirement API (extend `PatternContext` additively), `SegmentationMethod.segment`,
`PositionMover.move_to`, and readability of existing HDF5 files.

What not to do: move to real multiprocessing (nothing here needs it and the
code is shaped for threads); adopt a message bus or actor framework (the
queues are fine once they are `queue.Queue` and the topology is a table);
change file formats before the plan exists (the plan is what the writer
needs to know); or change the method API to pass docks/state explicitly
(keep the context object).

---

## 6. Verification performed for this assessment

- Read every file under `src/pyclm/` plus tests, sample TOMLs, docs, the
  unfinished commit `17523cd`, and the older feature branches.
- `uv run --group test pytest`: 8 passed in 62 s at HEAD.
- Scratch script: `multiprocessing.Queue.put/get` returns a different object
  with no shared memory (8 MB `uint16` round trip 6.6 ms); `queue.Queue`
  returns the same object; `empty()` immediately after `put()` returned
  `False` in the sampled run but is documented as unreliable.
- Scratch script: with h5py 3.14.0 / HDF5 1.14.6, modifying an existing
  attribute, creating a new attribute, and creating a dataset after
  `swmr_mode = True` all succeed without error.
- Dry-run reproduction of known-issues #1 with `t_delay = 1` (see that entry).
- Scratch script against useq-schema 0.8.1 confirming per-position
  sub-sequences, indexed events, and JSON round-trip (see seam A).
- `uvx ruff check --select F401` confirming the unused imports listed in
  known-issues #20; grep confirming only `microscope_to_manager` of the five
  `*_to_manager` queues is ever written to.
