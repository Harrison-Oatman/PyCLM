# Tracking

Tracking gives the objects your segmentation finds a stable identity across
timepoints, so a pattern method can follow a cell, an organoid or a tissue
region from one frame to the next, and so the output carries a per-object
table you can analyse afterwards. It is a stage of the closed loop like
segmentation: it runs live, on every acquired frame of a channel, and only
when a pattern method asks for it.

## Turning it on

Add a `[tracking]` table to the experiment TOML next to `[segmentation]`,
and ask for tracks in the pattern method:

```toml
[segmentation]
method = "cellpose"

[tracking]
method = "centroid"      # a built-in or registered tracking method
max_distance_um = 20     # remaining keys are passed to the method
save = true              # record the tracks in the output (default true)
# segmentation = "nuclei"  # track a named [segmentation.nuclei] table instead of the default

[pattern]
method = "follow_cells"
```

```python
class FollowCells(PatternMethod):
    name = "follow_cells"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_requirement("545", tracks=True)

    def generate(self, context):
        tracks = context.tracks("545")
        pattern = np.zeros(self.pattern_shape, dtype=np.float32)
        for row in tracks.rows:
            if row.track_id in self.chosen:        # ids persist across timepoints
                pattern[tracks.mask(row.track_id)] = 1.0
        return pattern
```

What happens: segmentation now runs on **every** acquired frame of channel
`545` (a tracker cannot link objects across frames it never saw), tracking
relabels each segmentation with track ids, and the pattern method receives
the tracked labels at its usual cadence. A `[tracking]` table that no
pattern method uses is a warning at start-up, not a running stage, in the
same way as an unused `[segmentation]`.

## What the pattern method sees

`context.tracks(channel)` returns a `Tracks` object for the current
timepoint:

| Member | Meaning |
|---|---|
| `tracks.labels` | the segmentation relabelled with track ids (`uint32`, 0 = background) |
| `tracks.rows` | one `TrackRow(track_id, label, y, x, area, parent)` per object; `y`, `x` are the centroid in pixels of the frame, `label` the id in the original segmentation, `parent` the parent track (0 if none) |
| `tracks.ids` | the track ids present, as an array |
| `tracks.mask(track_id)` | boolean mask of one object |
| `tracks.centroid(track_id)` | `(y, x)` or `None` if the id is not in this frame |
| `tracks.table`, `tracks.to_pandas()` | the rows as a pyarrow table / pandas frame |
| `tracks.measure(image)`, `tracks.paint(values)`, `tracks.select(ids)`, `tracks.areas()`, `tracks.centroids()` | inherited from `Regions`: the measurement toolbox in [custom pattern methods](custom_pattern_methods.md) |

`context.segmentation(channel)` still returns the untracked segmentation if
you asked for `seg=True` as well, and `context.history(channel,
kind="tracks", n=3)` gives the last three `Tracks` (see
[custom pattern methods](custom_pattern_methods.md)).

## Built-in tracking methods

**`centroid`** links each object to the nearest object of the previous
timepoint when its centroid moved less than `max_distance_um` (default 20),
with at most one partner per object (a Hungarian assignment on centroid
distance). Objects with no partner start a new track; objects that vanish
simply end. It does not detect divisions (`parent` is always 0) and does not
close gaps, so an object that is missed by segmentation for one frame comes
back with a new id. It is fast enough to be invisible in the loop.

For divisions and gap closing, write a method that wraps a tracker of your
choice (see below); [LapTrack](https://laptrack.readthedocs.io/), already
used in the lab's analysis, is a natural fit.

## Writing a tracking method

Subclass `TrackingMethod`, implement `track`, and register it:

```python
import numpy as np
from pyclm import TrackingMethod
from pyclm.core.tracking import TrackRow

class MyTracker(TrackingMethod):
    name = "my_tracker"

    def __init__(self, experiment_name, channel, max_jump_um=15.0, **kwargs):
        super().__init__(experiment_name, channel, **kwargs)
        self.max_jump_um = max_jump_um
        self.previous = None            # whatever state linking needs

    def track(self, labels, t, pixel_size_um):
        """labels: one label image (0 = background). Return it relabelled
        with track ids, plus one TrackRow per object in this frame."""
        ...
        return relabelled, rows
```

```python
run_pyclm(experiment_dir, tracking_methods={"my_tracker": MyTracker})
# or, with a Controller: c.register_tracking_method("my_tracker", MyTracker)
```

One instance is created per experiment and channel, with the keyword
arguments from the `[tracking]` table (everything except `method` and
`save`), and it receives every acquired frame of that channel in order.
Track ids should be positive integers that are never reused within a run.

## Where the tracks go

With the OME-Zarr format, the tracked labels are stored as a second label
image next to the segmentation (`imaging/labels/tracks/0`), and the rows go
to `tracks.parquet` (with `tracks.csv` at the end of the run) in the
experiment directory, one row per object and timepoint with the centroid in
pixels and micrometres. The ImageJ export adds the tracked labels as a
magenta channel after the segmentation. See [Data format](data_format.md).

The HDF5 format does not store tracks (it warns once and drops them);
segmentation and frames are unaffected.

```python
import pyclm.io

with pyclm.io.open("experiment_dir/bar10.pos1.zarr") as exp:
    g = exp.groups["imaging"]
    g.tracks(3, "545")            # tracked labels for slot 3, or None
    exp.tracks.to_pandas()        # every TrackRow of this experiment, with t and channel
```

## Two worked examples

Both methods below live in `documentation/examples/tracking_patterns.py`
and are exercised by the test suite, so they run as written. Register them
the usual way:

```python
from tracking_patterns import IntensityProgram, LeaderCells

run_pyclm(
    experiment_dir,
    pattern_methods={"leader_cells": LeaderCells, "intensity_program": IntensityProgram},
)
```

### Leader cells with directed stimulation

A fraction of the cells is chosen when the first pattern is generated and
keeps the counter-clockwise half-cell stimulus of `rotate_ccw` for the rest of the
movie. Nothing else is lit.

```{literalinclude} examples/tracking_patterns.py
:language: python
:pyobject: LeaderCells
```

```toml
[segmentation]
method = "cellpose"

[tracking]
method = "centroid"
max_distance_um = 15

[pattern]
method = "leader_cells"
channel = "545"
fraction = 0.2      # share of the cells present at the start that lead
seed = 0
```

How it works. The per-cell base classes (`PerCellPatternMethod`, which
`rotate_ccw` and the other movement methods build on, and
`NucleusControlMethod`) accept `tracks = true`. With it, their per-cell
loop runs on the tracked labels instead of the segmentation, so the region
label each method sees is a track id that persists across timepoints, and
the whole existing zoo of per-cell methods works on tracked cells with one
TOML key. `LeaderCells` only overrides `cell_labels`, the hook that
supplies that label image: on the first frame with cells it draws the
leaders with a seeded generator, and afterwards it blanks every region
whose id is not a leader before handing the image to the parent's loop.

Two consequences of tracking to keep in mind. A leader that the tracker
loses, because segmentation missed it for a frame or it moved further than
`max_distance_um`, comes back with a new id and stops being a leader; the
built-in `centroid` tracker does no gap closing, so tune the gate to the
motion you expect or use a tracker that does. And cells born after the
first frame are never leaders, which is the intended reading of "chosen at
the start of the movie".

### A three-phase intensity programme with per-cell feedback

Every cell receives no light for the first `dark_min` minutes and full
light for the next `light_min` minutes. From then on each cell is held at
the midpoint between the intensity it showed at the end of the dark phase
and at the end of the light phase.

```{literalinclude} examples/tracking_patterns.py
:language: python
:pyobject: IntensityProgram
```

```toml
[segmentation]
method = "cellpose"

[tracking]
method = "centroid"

[pattern]
method = "intensity_program"
channel = "545"
dark_min = 30
light_min = 30
gain = 2.0
```

How it works. The method asks for the raw frame and the tracks of the
channel (`add_requirement(channel, raw=True, tracks=True)`), so
`context.tracks` gives it lasting ids and `context.raw` the intensities.
`tracks.measure(raw)` is the mean intensity of every cell, one value per
`tracks.ids`. The phase is read from `context.time`, the scheduled seconds
since the run started. During the dark and light phases the method keeps
overwriting each cell's remembered level in a `PerTrack`, so what survives
is the value at the end of each phase, after the cells have settled. In the
feedback phase `PerTrack.get` reads the two levels back aligned with the
cells present, filling newcomers with the population median, and
`tracks.paint` turns the per-cell duty into the pattern. There is no
per-cell loop anywhere; the three helpers are described under the
measurement toolbox in [custom pattern methods](custom_pattern_methods.md).

The controller is proportional with a deliberately simple scale: a cell at
its target gets a 50 % duty pattern, one that has fallen a full
calibration span below it gets full light, one that has risen that far
above it gets none. Raising `gain` sharpens this towards the bang-bang
behaviour of `binary_nucleus_clamp`. Cells first seen after the light phase
have no calibration of their own and are held at the population medians.

Both examples keep their state on the method instance, as `fb_bounce` and
the cached embryo segmentation already do. For state that PyCLM should keep
for you, `context.history(channel, kind="raw", n=3)` returns the last three
raw frames delivered to the method, and `context.last_pattern()` the pattern
it produced last time; a controller with an integral or derivative term can
be written from those without any bookkeeping of its own.
