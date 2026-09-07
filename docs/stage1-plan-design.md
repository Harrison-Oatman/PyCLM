# Stage 1: the acquisition plan, and whether useq's `MDASequence` can be it

Date: 2026-09-06. Status: **implemented** the same day, after the five
decisions in §6 were confirmed (1: keep experiment-relative cadence; the user
notes `t_delay` has never been used in an experiment, so there was no
compatibility obligation either way; 2–5: confirmed as written). The code is
[`src/pyclm/core/plan.py`](../src/pyclm/core/plan.py); the tests are
[`tests/test_plan.py`](../tests/test_plan.py), which turned the spike's
Manager comparison into a permanent check against the scheduling rules. The
conversion spike script is therefore deleted; the feature probes remain in
[spikes/stage1_useq_probe.py](spikes/stage1_useq_probe.py). All results below
are from useq-schema 0.8.1, now a direct dependency in `pyproject.toml`.

---

## 1. Verdict

**Yes, with a thin PyCLM layer on top.** An `MDASequence` can carry the whole
PyCLM schedule (time plan, positions with per-position channel sets,
exposures, PFS offsets, delay/stop, stimulation cadence) as a declarative,
serialisable document, and PyCLM's per-timepoint event stream can be derived
from it. On the real dry-run test resources the derived stream is
**identical, event for event and in order, to what `Manager.process` emits
today**, both without and with `t_delay` (42/42 and 39/39 microscope events,
6/6 pattern requests).

What it cannot be is the executable event list on its own. Four things live
outside useq's model and must be added by a PyCLM wrapper:

1. **Cadence anchoring.** useq's `Channel.acquire_every` counts from absolute
   `t = 0`; PyCLM counts from the experiment's `t_delay`. Worse, useq 0.8.1
   ignores `acquire_every` on channels inside a position sub-sequence, which
   is where PyCLM's per-position channel sets have to live. So per-channel
   `every_t`, `t_delay` and `t_stop` must be applied by the wrapper.
2. **Non-acquisition events.** Pattern requests to the pattern process and
   the SLM update before each stimulation frame are PyCLM's own; useq has no
   way to insert them into a generated sequence (custom events can be built
   by hand, but not emitted by `iter_events`).
3. **Per-position time offset.** `time_between_positions` has no useq
   equivalent; the wrapper adds `p_index * between` to `min_start_time`.
4. **Hardware detail.** A useq event carries one `(group, config)` channel
   plus a flat `properties` list. PyCLM applies several config groups, device
   properties and binning per channel. Those stay in `ImagingConfig`, keyed by
   `(experiment, channel name)`, and the executor looks them up by identity.

None of these is an argument against useq: they are exactly the layer the
assessment called `AcquisitionPlan`, and it is small (the spike's
`derive_events` is 40 lines).

---

## 2. Why useq, and what it constrains

### Benefits of adopting useq for acquisition sequences

- **A standard, validated, serialisable experiment description.**
  `MDASequence` is a pydantic model: fields are typed, bad values fail at load
  time, unknown axes are rejected, and `yaml()` / `from_file()` round-trip
  exactly. PyCLM gets a plan file it can store next to the data, diff between
  runs, and reload, instead of a schedule that exists only as the Manager's
  loop.
- **The axes PyCLM wants next already exist.** Time (`TIntervalLoops`,
  `TDurationLoops`, `TIntervalDuration`, `MultiPhaseTimePlan`), z
  (`ZTopBottom`, `ZAboveBelow`, `ZRangeAround`, `ZAbsolutePositions`,
  `ZRelativePositions`), grids and multipoint (`GridRowsColumns`,
  `GridWidthHeight`, `GridFromEdges`, `GridFromPolygon`, `RandomPoints`,
  `WellPlatePlan`), per-position sub-sequences, and per-position autofocus
  plans. Z-stacks and tiled acquisition become plan changes plus storage
  work, not new scheduling code.
- **A canonical per-frame identity.** `MDAEvent.index` (`{t, p, g, c, z}`)
  is what the rest of the ecosystem keys on. pymmcore-plus ships writers that
  consume it directly (`OMEZarrWriter`, `OMETiffWriter`, `TensorStoreHandler`,
  `ImageSequenceWriter`), which gives Stage 2 a tested reference for the
  storage layout and a path to OME-Zarr if the lab wants it.
- **Interoperability with tooling PyCLM would otherwise build.**
  pymmcore-plus's `MDARunner.run()` accepts any iterable of `MDAEvent`s, and
  its engine exposes `setup_event` / `exec_event` / `teardown_event` hooks and
  hardware-triggered sequencing, so execution could later be delegated.
  pymmcore-widgets (not installed here; from its documentation) provides Qt
  editors for positions, channels, time, z and grid plans that read and write
  `MDASequence`, which is most of the Stage 5 interactive-setup UI.
- **Less bespoke code.** The Manager's loop and the four copies of "is
  channel X scheduled at t" collapse into "derive from the plan";
  `estimate_duration()` and `sizes` come for free, if partial.
- **Maintained upstream.** Schema evolution, validation and edge cases are
  handled by an active project; PyCLM tracks a version rather than owning a
  format.

### Built-in restrictions of `MDASequence`

- **Events are a Cartesian product of per-axis plans**, nested in
  `axis_order` (`t ⊃ p ⊃ g ⊃ c ⊃ z` by default). Anything that is not "every
  combination of these plans" needs a sub-sequence or a post-filter.
  Irregular per-channel time sets (channel B at t = 5, 7, 11) cannot be
  written down.
- **One clock for the whole sequence.** The time plan is global; a
  sub-sequence time plan nests inside each outer timepoint rather than
  overriding it (verified). Per-position delays (`t_delay`), early stops
  (`t_stop`), different intervals per position, and per-position offsets are
  not expressible. `MultiPhaseTimePlan` only changes the interval globally.
- **Cadence is regular, absolute, and integer.** `Channel.acquire_every = n`
  means "every n-th timepoint counted from t = 0"; there is no phase, no
  per-position anchoring, and in 0.8.1 it is ignored inside position
  sub-sequences.
- **All plans are bounded.** Every time plan needs `loops` or `duration`;
  there is no "run until stopped". Open-ended experiments need a large loop
  count and a stop command from outside.
- **The sequence is static and immutable.** `MDASequence` and `MDAEvent` are
  frozen pydantic models; changing anything means building a new sequence
  with `replace()`. There is no conditional or feedback-driven event
  generation ("if cells are found, take a z-stack", "re-image if focus
  fails"). For PyCLM this matters less than it sounds, because the closed
  loop acts on the *pattern*, not on the acquisition; but Stage 4 runtime
  edits must be implemented as "swap in a new plan at a timepoint boundary",
  and the executor must tolerate that.
- **One `(group, config)` per event, no per-channel metadata.** `Channel` has
  `config`, `group`, `exposure`, `do_stack`, `z_offset`, `acquire_every`,
  `camera` and nothing else; an event has one `channel` and a flat
  `properties` list. PyCLM's stacked config groups (Channel + LightPath +
  Laser-Intensity), per-channel device properties, binning and `save` flags
  must travel outside the channel, keyed by name.
- **Positions are `x, y, z, name, row, col` plus a sub-sequence.** No extras
  dict. The PFS offset fits the autofocus plan; any other per-position
  hardware value has to go in `metadata`.
- **Stimulation is not a concept.** useq models imaging. `MDAEvent.slm_image`
  exists as a per-event payload, but nothing in plan generation says "update
  the SLM, then acquire", and pattern-generation requests are entirely
  outside the model. PyCLM must synthesise those events itself, which the
  spike does.
- **Z-plans are per sequence or per position, not per channel.** A channel
  can opt out (`do_stack=False`) or shift (`z_offset`), but two channels
  cannot have different z ranges at one position without a second
  sub-sequence.
- **Timing is advisory.** `min_start_time` is a minimum; the runner decides
  pacing. Deadlines, the setup phase and between-position gaps are the
  executor's business, and `estimate_duration()` counts exposures only.
- **`metadata` is where flexibility comes back, at a cost.** It is a free
  dict: unvalidated, invisible to pymmcore-widgets, and easy to let drift
  from the fields around it. Everything PyCLM puts there (`every_t`,
  `t_delay`, `t_stop`, the stimulation channel, binning) needs its own
  validation.
- **Version coupling.** useq is 0.x; behaviour such as sub-sequence
  `acquire_every` can change between releases. Stored plans should record
  the useq version, and PyCLM semantics should never depend on a useq
  behaviour the wrapper does not also enforce.

### What this means for PyCLM

The restrictions all fall on the same side: useq is a good description of
*what to acquire and when, regularly*; it is not a description of *how PyCLM
reacts*. The design in §5 keeps the sequence as the declarative core and puts
everything reactive or PyCLM-specific in `metadata["pyclm"]` plus the
`AcquisitionPlan` wrapper. The price is that two sources describe one
experiment; the guards against drift are that the wrapper, not useq, is the
authority for cadence and timing (decision 1 in §6), and that the spike's
Manager comparison becomes a permanent test.

---

## 3. What was verified

| Question | Result |
|---|---|
| Per-position channel sets with different lists per position | Works (`Position.sequence=MDASequence(channels=[...])`). |
| Per-channel cadence via `Channel.acquire_every` | Works at top level; **silently ignored inside a position sub-sequence** (probe: `545` with `acquire_every=2` still fires every `t`). |
| `t_delay`/`t_stop` via a sub-sequence `time_plan` | Does not work: a sub-sequence time plan is a nested inner loop run at every outer timepoint, not an override. |
| PFS offset per position | `AxesBasedAF(autofocus_device_name="PFS", autofocus_motor_offset=..., axes=("p",))` on the position's sub-sequence emits one `HardwareAutofocus` event on arrival at each position. |
| Stimulation as a channel | Works as `Channel(config="DMD", group="Channel")`, listed first so it precedes imaging channels. `do_stack=False` keeps it to one event per timepoint under a z-plan (probe: DMD once at the middle z, `545` three slices). |
| Z-stack on one position only | Works; events carry `index["z"]` and `z_pos`; the storage path derives directly (`00000/channel_545/z02`). |
| Serialisation | `seq.yaml()` / `MDASequence.from_file()` round-trip equal; 1.4 kB for two positions. |
| Scale | 1440 timepoints × 12 positions × 3 channels = 51,839 events enumerate in 0.96 s (19 µs/event); grouping by `t` into a dict takes 1.25 s. Building the whole plan at start-up is fine. |
| `estimate_duration()` | Counts exposures only (1.2 s per timepoint for 36 events). PyCLM's budget is dominated by settle time and stage moves, so it needs its own estimate on top of the plan. |
| Feeding events to pymmcore-plus later | `MDARunner.run(events: Iterable[MDAEvent])` accepts any iterable; `MDAEngine` exposes `setup_event`/`exec_event`/`teardown_event`. Left as a later option; Stage 1 keeps PyCLM's executor. |
| `sizes`/`shape` | Only meaningful for top-level axes. With per-position sub-sequences they report `c: 0, z: 0`; consumers must derive shapes by enumeration. |

---

## 4. Mapping

| PyCLM today | In the plan |
|---|---|
| `schedule.toml` `steps`, `interval_seconds` | `time_plan = TIntervalLoops(interval, loops)` |
| `setup_time_seconds`, `time_between_positions` | `MDASequence.metadata["pyclm"]` |
| One `Experiment` per position label | `Position(x, y, z, name=label, sequence=...)` |
| `PositionWithAutoFocus` / `extras["PFSOffset"]` | `autofocus_plan=AxesBasedAF(...)` on the position sub-sequence |
| `[channels].group`, `presets`, per-channel `exposure` | `Channel(config=preset, group=group, exposure=...)` in the sub-sequence |
| `[stimulation]` | First `Channel` of the sub-sequence, named by its preset in the channel group (`"DMD"` in the lab configs), `do_stack=False`; name recorded in metadata as `stim_channel` |
| Per-channel `every_t`, `t_delay`, `t_stop`, `pattern.every_t`, binning | `Position.sequence.metadata["pyclm"]` (source of truth; `acquire_every` left at 1, see §6) |
| Extra config groups, device properties, `save` flags | Stay in `ImagingConfig`/`Experiment`, looked up by `(experiment, channel)` at execution |
| `AcquisitionEvent.sub_axes = [t, "channel_x"]` | `index = {"t", "p", "c"[, "z"]}` from the useq event; path derived from it |
| Pattern request cadence (`lcm`) | Computed by the wrapper from metadata + requirements, as `Manager.get_pattern_lcm` does now |
| `UpdatePatternEvent`, `UpdateStagePositionEvent`, `RequestPattern` | Emitted by the wrapper around the useq events |

---

## 5. Proposed design

### `pyclm/core/plan.py`

```python
class AcquisitionPlan:
    sequence: useq.MDASequence            # declarative, serialisable
    schedule: ExperimentSchedule          # hardware detail by (experiment, channel)

    @classmethod
    def from_schedule(cls, schedule) -> "AcquisitionPlan"      # schedule_to_mda
    @classmethod
    def from_yaml(cls, path, schedule) -> "AcquisitionPlan"
    def to_yaml(self, path)                                     # provenance

    @property
    def timepoints(self) -> int
    def events_at(self, t) -> list[PlannedEvent]                # request_pattern, position,
                                                                # update_pattern, acquire; in order
    def is_scheduled(self, experiment, channel, t) -> bool      # the one predicate
    def expected_datasets(self) -> Iterator[tuple[experiment, index, kind]]
    def estimate_timepoint_s(self, t, settle_s, move_s) -> float
```

`PlannedEvent` is a frozen dataclass: `kind`, `t`, `experiment`, `channel`,
`index`, `scheduled_offset_s`, `needs_slm`, `z_pos`, plus `make_pattern` /
routing flags resolved from the requirements. The Manager turns each one into
the existing message types, so the microscope, outbox, segmentation and
pattern processes are untouched in Stage 1.

### `AcquisitionEvent.index`

Replace `sub_axes` (a list of strings) with `index: dict[str, int | str]`
carrying `t`, `p` (experiment name), `c` (channel name) and later `z`/`g`.
`get_rel_path()` derives today's `00012/channel_545/` from it, so the HDF5
layout is unchanged; `write_attrs` records the index. Dock keys in the pattern
process become `(experiment, t)` from the index rather than a formatted
string.

### Who consumes the plan

- `Manager.process`: `for t in range(plan.timepoints): wait; for ev in plan.events_at(t): send`. `get_kwargs`, `get_pattern_lcm`, `send_make_pattern_request` and the timepoint gating move into the plan.
- `MicroscopeOutbox.initialize` and `_timepoint_complete`: iterate `plan.expected_datasets()` / call `plan.is_scheduled`. Removes two of the four duplicated predicates (known-issues #14).
- `run_pyclm`: writes `plan.useq.yaml` into the experiment directory and stores the same YAML in each HDF5 file's root attrs; prints the timing estimate and warns when a timepoint's estimated duration exceeds the interval.
- The GUI keeps its own `ChannelSchedule` in Stage 1 (it is a separate process reading files; Stage 2 hands it the plan via the file).

### Tests

- The spike's comparison becomes a permanent test: `plan.events_at(t)` over all `t` equals the Manager's emission on the test resources and on the `tests/helpers.py` schedules, across `t_delay`, `t_stop`, `every_t`.
- YAML round trip; `index` → path; `expected_datasets()` matches `test_dry_run.py`'s inventory; a z-stack plan enumerates `z` for imaging channels only; the timing estimate flags an over-full interval.

Rough size: 1 to 2 weeks, of which the plan class and its tests are most of it;
the Manager and Outbox edits are subtractions.

---

## 6. Decisions to confirm before implementing

1. **Cadence stays experiment-relative** (`(t - t_delay) % every_t == 0`),
   as today. Consequence: `Channel.acquire_every` is left at 1 in the
   sequence so a future useq version that honours it in sub-sequences cannot
   silently change behaviour; `every_t` lives in `metadata["pyclm"]`. The
   alternative, adopting useq's absolute anchoring, would shift the phase of
   every delayed experiment with `every_t > 1`. Recommendation: keep today's
   semantics.
2. **The stimulation channel's name** in the plan is its preset in the
   channel group (`"DMD"`), falling back to `"stimulation"` when the
   stimulation config has no entry in that group. Storage keeps `stim_aq` as
   the group name in Stage 1; renaming is a Stage 2 layout question.
3. **PFS offsets** go into the plan as an `AxesBasedAF` for provenance and
   for pymmcore-widgets, but Stage 1 does not send the resulting
   `HardwareAutofocus` events to the microscope: `PositionMover` keeps owning
   focus. Switching to AF events is a later, separate change.
4. **`plan.useq.yaml` is written next to the HDF5 files** and embedded in
   their root attributes. This is the first artefact a Stage 5 setup GUI can
   load and edit.
5. **Scope guard.** Stage 1 does not add z or grid support; it makes the plan
   the single place they will be added. The z-stack probe above is evidence
   the representation is ready, not a feature.
