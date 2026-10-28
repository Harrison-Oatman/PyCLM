# Gap closing in live tracking: design note

Status: requested by Harrison on 2026-10-28 ("pyclm tracking methods do not
currently allow gap closing ... We can also consider adding a dependency to
some tracking package that works efficiently in python, to avoid
reimplementing what has already been optimized"). Also item 22 of the
session 3 recommendations. Harrison approved every decision in §4 as
recommended on 2026-10-28; implemented the same day (§6).

## 1. Verdict

Write the gap closing ourselves, inside the built-in `centroid` tracker, in
about a hundred lines of numpy/scipy (both already dependencies). None of
the packages fits a live loop: laptrack and btrack are offline trackers
(they link a whole movie at once), and trackpy, the one package with a
frame-by-frame linker, raises an exception and stops linking when a field
is dense or the cells move far between frames. A prototype of our own
linker bridges as many gaps as trackpy, never fails, and costs 3 to 40 ms
per frame for 200 to 3000 nuclei, which is less than the label-image work
the tracker already does every frame (35 to 120 ms, §3.3). laptrack stays
where it is, in the `analysis` group, for retracking a finished run with
divisions.

## 2. What we have

- `CentroidTracker.track` (`src/pyclm/core/tracking/centroid.py:45`) links
  each region to the nearest region of the **previous frame only**, with a
  dense Hungarian assignment on all centroid pairs
  (`linear_sum_assignment(distance)`, `centroid.py:61`), then drops pairs
  farther than `max_distance_um`. An object that segmentation misses for
  one frame is forgotten and comes back with a new id.
- The tracking process (`src/pyclm/core/tracking_process.py:88`,
  `handle_data`) calls `method.track(labels, t, pixel_size_um)` once per
  acquired frame of the channel, in order, in its own thread, and calls
  `method.shift` first when the stage moved (`_follow_stage`, line 124).
- The interface (`TrackingMethod`, `TrackRow` in `tracking.py`) has room for
  what gap closing needs: ids must be positive and never reused, and an
  object absent from a frame simply has no row.
- What users see today, documented in `documentation/tracking.md` ("Two
  consequences of tracking to keep in mind"): a leader cell that
  segmentation misses for one frame loses its leader status; a `PerTrack`
  memory restarts from the population median.

The live constraint that decides everything below: **an id handed to a
pattern method cannot be taken back.** The light has already been
projected on that cell. A tracker that revises past links after seeing
later frames (the global LAP of Jaqaman et al., which is what laptrack and
btrack's optimiser do) gives the right answer only after the fact. Live,
the tracker must decide at each frame from what it has seen so far.

## 3. What was measured

Spike: [spikes/tracking_gap_closing_bench.py](spikes/tracking_gap_closing_bench.py)
and [spikes/tracking_btrack_probe.py](spikes/tracking_btrack_probe.py), run
2026-10-28 in the cloud container (4-core Xeon), Python 3.12, numpy 2.5,
scipy 1.18, trackpy 0.7, laptrack 0.17.1, btrack 0.7.0.

Synthetic nuclei in a 2048 × 2048 frame, 60 frames: a random walk with a
per-axis step of σ pixels plus a small drift; each frame 5 % of the objects
are missed by "segmentation", 40 % of those for two frames; 0.5 % new
objects per frame. Gate = 5σ, gaps of up to 2 frames bridged. Only the
linking is timed. "Links kept" is the fraction of consecutive-frame links
that keep their id; "gaps bridged" the fraction of objects that come back
after a gap with their old id; "id swaps" counts ids that move to another
true object (merges and swaps; partly unavoidable at high density).

### 3.1 Latency and accuracy

| linker | objects | σ (px) | median / p95 ms | links kept | gaps bridged | id swaps |
|---|---|---|---|---|---|---|
| `centroid` today | 200 | 3 | 0.6 / 0.8 | 0.964 | 0.000 | 27 |
| own prototype | 200 | 3 | 3.4 / 5.1 | 0.999 | 0.947 | 15 |
| trackpy `Linker` | 200 | 3 | 0.7 / 0.9 | 0.999 | 0.947 | 13 |
| laptrack, 6-frame window | 200 | 3 | 158 / 186 | – | – | – |
| `centroid` today | 1000 | 3 | 17 / 30 | 0.923 | 0.015 | 933 |
| own prototype | 1000 | 3 | 16 / 21 | 0.991 | 0.938 | 637 |
| trackpy `Linker` | 1000 | 3 | 4.8 / 34 | 0.990 | 0.937 | 642 |
| laptrack, 6-frame window | 1000 | 3 | 844 / 988 | – | – | – |
| `centroid` today | 3000 | 3 | 232 / 381 | 0.876 | 0.037 | 8554 |
| own prototype (penalty 0.1 gate) | 3000 | 3 | 40 / 45 | 0.973 | 0.866 | 5427 |
| trackpy `Linker` | 3000 | 3 | 20 / 60 | 0.974 | 0.894 | 5292 |
| laptrack, 6-frame window | 3000 | 3 | 3505 / 4237 | – | – | – |
| own prototype (penalty 0.1 gate) | 1000 | 10 | 7.8 / 8.4 | 0.907 | 0.710 | 6278 |
| trackpy `Linker` | 1000 | 10 | **fails at frame 2**: `SubnetOversizeException` | | | |
| own prototype (penalty 0.1 gate) | 3000 | 6 | 21 / 22 | 0.900 | 0.695 | 20322 |
| trackpy `Linker` | 3000 | 6 | **fails at frame 1**: `SubnetOversizeException` | | | |

The 200- and 1000-object prototype rows are with no gap penalty (§4,
decision 4); with a penalty of 0.1 × gate per missed frame the prototype
bridged 0.922 and 0.907 of the gaps instead.

### 3.2 The candidates

- **laptrack** (BSD, pure Python, last release 2026-02, already in the
  `analysis` group). The best tool for analysis: the full Jaqaman LAP with
  gap closing, splitting and merging. It is offline: `predict` takes the
  whole list of frames and returns a graph, and earlier links can change
  when frames are added. Re-running it on a sliding window to get the
  current frame's ids costs 0.16 to 3.5 s per frame (table above), and ids
  would still have to be carried from window to window by hand. Not for
  the live loop.
- **btrack** (MIT, ships a compiled `libtracker` for Windows, macOS and
  Linux inside a pure-Python wheel; 0.7.0, 2025-09; pulls in cvxopt and
  h5py). A Bayesian (Kalman) tracker whose gap closing and divisions come
  from a global hypothesis optimisation (`optimise`) run after tracking.
  The probe appended one frame of objects and called `track(step_size=1)`
  per frame: after the first call btrack reports "No trackable frames have
  been found" and links nothing further, so it does not track
  incrementally. Not for the live loop.
- **trackpy** (BSD, pure Python, sdist only, last release 0.7 on
  2025-07; depends on pandas, matplotlib, pyyaml). Its `trackpy.linking.Linker`
  links frame by frame (`init_level`, `next_level`) and closes gaps with
  `memory` (an unmatched particle is kept for `memory` frames at its last
  position). It is as accurate as our prototype and a little faster at the
  median, but it solves each ambiguous cluster ("subnet") by enumeration
  and raises `SubnetOversizeException` when a cluster has more than 30
  particles. That happened in the first frames of every dense or fast case
  tried. Its remedy, `adaptive_stop`, silently shrinks the search range for
  that cluster. A live run must not stop tracking because a field filled
  up, and catching the exception means losing that frame's links.
  `Linker` is also a lower-level class than trackpy's documented `link`
  functions.
- **Our own linker** (prototype `GapLAP` in the spike). Active tracks and
  tracks unseen for up to `max_gap` frames are both candidates for each new
  detection. Pairs within the gate are found with a KD-tree
  (`scipy.spatial.cKDTree`), split into connected clusters, and each
  cluster is solved as one assignment problem with a "no link" option at
  the gate cost for every track and every detection (the augmented matrix
  of Jaqaman et al. 2008, the frame-to-frame half of what laptrack does).
  It never fails, its cost is bounded by cluster size rather than field
  size, and it is faster than today's dense Hungarian above ~1000 objects
  (40 vs 232 ms at 3000). It uses only scipy and numpy.

### 3.3 Where the time goes

The tracker's other per-frame work, `regionprops_table` for centroids plus
the relabelling lookup on a 2048 × 2048 label image, measured 34, 62 and
122 ms for 200, 1000 and 3000 objects. Linking with gap closing adds 3 to
40 ms to that, and Cellpose itself takes seconds. Tracking runs in its own
thread, once per acquired frame per position, so the sum over positions
must fit in the interval; with 20 positions of 1000 cells that is about
1.6 s per timepoint, comparable to today. Latency does not separate
trackpy from our own linker; robustness and dependencies do.

## 4. Decisions to confirm

1. **Own implementation, no new dependency.** Recommended, for the reasons
   in §3.2: the offline packages do not fit the live loop and trackpy stops
   linking in dense or fast fields. Alternative: trackpy's `Linker` with a
   fallback to "new ids for that cluster" on `SubnetOversizeException`
   (adds trackpy, pandas and matplotlib to the core install, which has
   neither pandas nor matplotlib today).
2. **Gap closing goes into `centroid`, not a new method name.** One tracker
   to explain; `max_gap_frames = 0` reproduces today's links exactly except
   where §4.6 changes the solver. A new name (`lap`) would leave users
   choosing between two near-identical trackers.
3. **`max_gap_frames`, default 2.** The number of consecutive frames of the
   tracked channel an object may be missing and still keep its id. Counted
   in frames, not minutes, which departs from K1 of the keyword convention
   (`pattern-methods-expansion-design.md` §2): a gap is a number of missed
   segmentations, and a tracked channel at `every_t = 3` would otherwise
   need `max_gap_min` divisible by its own interval. The default turns gap
   closing on for existing experiments; that is the behavior the leader-cell
   and `PerTrack` examples want. Alternative: default 0 (today's behavior),
   opt in.
4. **No penalty for missed frames; the same `max_distance_um` gate for gap
   links.** The measured penalty of 0.1 × gate per missed frame lost 3 to
   4 % of the bridged gaps and did not reduce swaps. A gate that grows with
   the gap (√(gap + 1) for diffusive motion) is an easy later addition if a
   scope run shows cells leaving the gate during a gap.
5. **A missing track waits at its last seen position, without a motion
   prediction**, and `shift` moves it like an active track, so ids survive
   a stage move during a gap. A constant-velocity prediction can come later
   if migrating cells outrun the gate.
6. **Replace the dense Hungarian with the gated, clustered assignment for
   every link**, not only for gap closing. It is the correct optimum (today
   the solver may pair two far-apart objects and then drop the pair, which
   can displace a valid link) and it is 6× faster at 3000 objects. Links
   can differ from today's in crowded clusters; the tests compare against
   ground-truth synthetic tracks rather than against today's output.
7. **No change to the output.** The tracks table has no row for a frame an
   object was missed in; the id is the same before and after. No `gap`
   column and no interpolated rows: the frames table already says which
   frames were acquired, and interpolating positions belongs in analysis.
8. **Divisions stay out of scope.** `parent` stays 0. A live division rule
   (a new object next to a track that just shrank) can be a later note;
   for analysis, `documentation/tracking.md` gets a short recipe for
   retracking `tracks.parquet` with laptrack (splitting on), which needs no
   new dependency in the core.

## 5. What changes if approved

- `src/pyclm/core/tracking/centroid.py`: the linker (§3.2, last item) with
  `max_gap_frames`; `shift` moves waiting tracks too.
- Tests (`tests/`): synthetic tracks with dropped frames (bridged within
  `max_gap_frames`, new id after it), a stage shift during a gap, a dense
  field that must not fail, `max_gap_frames = 0` behaving like today on
  uncrowded data, ids never reused.
- `documentation/tracking.md`: the new keyword, the paragraph on missed
  frames in the leader-cell example, the laptrack retracking recipe.
- `pyclm check` validates the new keyword through the constructor signature
  as it does today; nothing to add there.
- CHANGELOG under Unreleased; CLAUDE.md and the `pyclm-write-pattern-method`
  skill only if the tracking surface they describe changes (it does not,
  beyond the keyword).

Testing at the microscope: a nuclear-marker run with tracking on and a
`PerTrack`-based method, comparing the number of track ids per position
with `max_gap_frames = 0` and `2` (same TOML, two positions), and the
tracking time per frame in the log.

## 6. Implementation notes

- `src/pyclm/core/tracking/centroid.py`, `link()`: the prototype solved
  each connected cluster with its own `linear_sum_assignment`. The shipped
  version builds the same augmented matrix once, as a sparse matrix, and
  solves it with one `scipy.sparse.csgraph.min_weight_full_bipartite_matching`
  call (costs carry a 1e-9 offset because a sparse matrix drops explicit
  zeros). On the spike's data it gives identical links, gap bridging and
  swaps, and is 5 to 8 times faster: 0.7, 1.9 and 5.7 ms median for 200,
  1000 and 3000 nuclei (σ 3 px), 2.6 ms for 1000 at σ 10 px; measured
  with a scratch script reusing `simulate` and `score` from the spike.
- The gap is counted in calls to `track`, i.e. in frames of the tracked
  channel, so a channel imaged every third timepoint counts its own frames.
- Tests: `tests/test_tracking.py` (gap bridged within `max_gap_frames`,
  forgotten after it, `max_gap_frames = 0`, a waiting and a present track
  competing for one object, a stage shift during a gap, a dense field of
  400 objects with 10 % missed per frame, ids never reused, a negative
  gap rejected).
