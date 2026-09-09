# Writing Custom Pattern Methods

## Overview
Custom patterns provide a way to extend PyCLM and enable a wide range of new experiments using python.
The custom pattern API is set up to minimize the amount of boilerplate code the user has to write.

The first step in writing a custom pattern method is to implement a subclass of the `PatternMethod` base class.

```python
from pyclm import PatternMethod

class MyCustomPattern(PatternMethod):
    """A custom pattern method"""
    ...
```

The subclass needs to implement two methods — `__init__` and `generate` — and set a `name` class attribute.

```python
from pyclm import PatternMethod, PatternContext

class MyCustomPattern(PatternMethod):
    """A custom pattern method."""

    name = "custom_method"  # used in logging; does not need to match the run_pyclm dict key

    # keyword arguments defined here can be set in the experiment .toml
    def __init__(self, keyword_a="default_value_a", keyword_b=42, **kwargs):
        super().__init__(**kwargs)  # boilerplate, always needs to be present

        # store keywords as attributes so generate() can access them
        self.attribute_a = keyword_a
        self.attribute_b = keyword_b

        # declare what imaging data generate() needs — called once at startup
        self.add_requirement("RFP", raw=True, seg=False)   # raw image from the RFP channel
        self.add_requirement("GFP", raw=False, seg=True)   # segmentation mask from the GFP channel

    def generate(self, context: PatternContext):
        # unpack data declared in __init__ via add_requirement()
        raw = context.raw("RFP")           # np.ndarray — most recent RFP image
        seg = context.segmentation("GFP")  # np.ndarray — labelled segmentation mask

        # x/y coordinates of every pixel in microns (same shape as the camera output)
        xx, yy = self.get_um_meshgrid()

        # center of the field of view in microns
        cx, cy = self.center_um()

        # return a float array in [0, 1] the same shape as the camera output
        # e.g. illuminate the right half at the intensity fraction set by keyword_b:
        pattern = (self.attribute_b / 100) * (xx > cx)

        return pattern
```

When you run pyclm, you must supply any custom generated patterns in a dictionary, with the name of the method 
as it should be referenced in the .toml

```python
from pyclm import run_pyclm
from my_pattern import MyCustomPattern

experiment_directory = ...

run_pyclm(experiment_directory, pattern_methods={"custom_method": MyCustomPattern})
```

PyCLM will know that you want to use this pattern if you put it in your .toml. The .toml should also supply any keyword 
arguments that you want to overwrite.
```toml
# experiment_a.toml

[pattern]
method = "custom_method"
keyword_b = 75  # illuminate the right half at 75% intensity
```



## Example 1 — static (open-loop) pattern

This method illuminates an annulus whose inner and outer radii are set in the TOML. It requires no live imaging data.

```python
# my_patterns.py
import numpy as np
from pyclm import PatternMethod


class AnnulusPattern(PatternMethod):
    """Illuminates a ring centerd on the FOV."""

    name = "annulus"

    def __init__(self, inner_radius_um=30, outer_radius_um=60, **kwargs):
        super().__init__(**kwargs)
        # No add_requirement calls — this pattern needs no image data.
        self.r_inner = inner_radius_um
        self.r_outer = outer_radius_um

    def generate(self, context) -> np.ndarray:
        xx, yy = self.get_um_meshgrid()
        cx, cy = self.center_um()

        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

        ring = (dist_sq >= self.r_inner ** 2) & (dist_sq <= self.r_outer ** 2)
        return ring.astype(np.float32)
```

**Matching TOML:**

```toml
[pattern]
method = "annulus"
inner_radius_um = 25
outer_radius_um = 80
```

All keys other than `method` are passed verbatim as kwargs to `__init__`. Default values in `__init__` are used when a key is absent from the TOML.

**Registration:**

```python
from pyclm import run_pyclm
from my_patterns import AnnulusPattern

run_pyclm(
    experiment_directory,
    pattern_methods={"annulus": AnnulusPattern},
)
```

---

## Example 2 — feedback-controller (closed-loop)

This method reads per-cell mean intensity and illuminates only the cells below a target, with the threshold tunable from the TOML.

```python
# my_patterns.py
import numpy as np
from skimage.measure import regionprops
from pyclm import PatternMethod


class IntensityThresholdPattern(PatternMethod):
    """Illuminates cells whose mean fluorescence is below `target_intensity`."""

    name = "intensity_threshold"

    def __init__(self, channel="GFP", target_intensity=3000, **kwargs):
        super().__init__(**kwargs)

        self.channel = channel
        self.target = target_intensity

        # Declare that generate() needs both the raw image and the
        # segmentation mask for the chosen channel.
        self.add_requirement(channel_name=channel, raw=True, seg=True)

    def generate(self, context) -> np.ndarray:
        raw = context.raw(self.channel)          # float array, camera coords
        seg = context.segmentation(self.channel) # label array, same shape

        h, w = self.pattern_shape
        out = np.zeros((int(h), int(w)), dtype=np.float32)

        for prop in regionprops(seg, intensity_image=raw):
            if prop.intensity_mean < self.target:
                r0, c0, r1, c1 = prop.bbox
                out[r0:r1, c0:c1] += prop.image  # binary mask of this cell

        return np.clip(out, 0, 1)
```

**Matching TOML:**

```toml
[segmentation]
method = "cellpose"
model = "cpsam"

[pattern]
method = "intensity_threshold"
channel = "GFP"
target_intensity = 4500
```


---

## Base class API

```python
from pyclm.core.patterns.pattern import PatternMethod, PatternContext
```

| Method / attribute | Purpose |
|---|---|
| `self.pattern_shape` | `(height, width)` in pixels, set by `configure_system` |
| `self.pixel_size_um` | microns per pixel (accounts for binning) |
| `self.get_um_meshgrid()` | returns `(xx, yy)` arrays in µm, shape `pattern_shape` |
| `self.center_um()` | returns `(cx, cy)` center of the FOV in µm |
| `self.add_requirement(channel_name, raw=False, seg=False, tracks=False, history=1)` | declare that `generate` needs the raw frame, the segmentation and/or the tracks of a channel, and how many past deliveries to keep. `seg` is `True` for the default `[segmentation]` table, the name of a `[segmentation.<name>]` table, or a list of names |
| `self.request_stim(raw=False, seg=False, history=1)` | same, but for the stimulation-output channel |
| `pattern_history` (class attribute, default 2) | how many of the method's previous patterns the context keeps |

The `generate` method receives a `PatternContext` and must return a `float` array with values in `[0, 1]` and shape matching `self.pattern_shape`.

```python
context.raw(channel_name)           # np.ndarray – raw fluorescence image
context.segmentation(channel_name)  # np.ndarray – labelled segmentation mask
context.segmentation(channel_name, "nuclei")   # the labels of a named [segmentation.nuclei] table
context.regions(channel_name, name="segmentation")  # Regions: measure(), paint(), ... (toolbox below)
context.tracks(channel_name)        # Tracks – tracked labels + per-object rows (see Tracking)
context.stim_raw()                  # raw image of the stimulation channel
context.stim_seg()                  # its segmentation
context.time                        # elapsed experiment time in seconds
context.t                           # plan timepoint (int)
context.generation                  # how many patterns this method has produced so far

context.history(channel_name, kind="seg", n=None)   # past deliveries, oldest first, current last
context.stim_history(kind="raw", n=None)
context.last_pattern()              # the array generate() returned last time, or None
context.pattern_history(n=None)     # previous patterns, oldest first

context.settings(channel_name)      # exposure, presets, device properties in force
context.set_exposure(channel_name, ms)                 # change settings of this experiment
context.set_property(channel_name, device, prop, value)  # from the next timepoint on
```

History is sampled at the pattern's own cadence: `add_requirement("545",
seg=True, history=3)` keeps the last three segmentations that reached this
method, which is what a controller that integrates or differentiates its
input needs. Only what you asked for is kept, so memory stays bounded.
Anything that must see *every* acquired frame of a channel belongs in a
tracking method, not in `generate` (see [Tracking](tracking.md)).

---
## The measurement toolbox

Most per-cell methods do the same three chores: measure something per
cell, remember something per cell, and turn per-cell numbers back into a
pattern. PyCLM provides them, so a method reads like the experiment it
runs rather than like image-processing plumbing.

```python
from pyclm import Regions, PerTrack, nuclear_cytosolic_ratio

regions = context.regions("545")            # a segmentation as Regions
tracks = context.tracks("545")              # a Tracks is a Regions whose ids persist

regions.ids                                 # object ids, ascending (tracks: in row order)
regions.measure(image)                      # mean of image per object, aligned with ids
regions.measure(image, "max")               # also median, min, sum, std, var
regions.areas(), regions.centroids()        # pixels per object; (N, 2) centroids y, x
regions.paint(values)                       # an image with each object filled with its value
regions.paint(0.5)                          # ... the same value for every object
regions.paint({12: 1.0})                    # ... or a dict by id
regions.select(ids)                         # the same frame keeping only those objects
cells.owner_of(nuclei)                      # for each nucleus, the cell under its centroid

memory = PerTrack()                         # a value remembered per track id
memory.update(tracks.ids, values)           # the latest value wins
memory.get(tracks.ids, default="median")    # aligned with the ids; unknown ids get the default
```

With these, the feedback phase of a per-cell controller is a few lines (the
full method is in [Tracking](tracking.md)):

```python
means = tracks.measure(context.raw(self.channel))
low = self.low.get(tracks.ids, default="median")
high = self.high.get(tracks.ids, default="median")
duty = 0.5 + self.gain * ((low + high) / 2 - means) / np.maximum(high - low, 1e-6)
return tracks.paint(np.clip(duty, 0, 1))
```

`paint` is also the fast path for any per-cell pattern: it fills every
object in one array operation instead of looping over regions.

---

## Example 3 — two segmentations of one channel (a KTR clamp)

A kinase translocation reporter (KTR) is read out as the ratio of nuclear
to cytosolic intensity of one channel, which needs two segmentations of
that channel: nuclei and whole cells. Name them in the TOML:

```toml
[segmentation.nuclei]
method = "cellpose"
model = "nuclei"

[segmentation.cells]
method = "cellpose"
model = "cyto3"

[pattern]
method = "ktr_clamp"
channel = "ktr"
target = 1.2
gain = 2.0
```

and ask for both by name:

```{literalinclude} examples/ktr_patterns.py
:language: python
:pyobject: KTRClamp
```

`nuclear_cytosolic_ratio` matches every nucleus to the cell under its
centroid, measures the reporter inside the nucleus and in the cell minus
every nucleus, and returns the ratio per nucleus together with the cell it
belongs to, so the pattern can be painted on whole cells. A nucleus outside
any cell gets `NaN` and no light. With a nuclear marker channel instead,
segment that channel for the nuclei (`add_requirement("nuc", seg=True)`)
and the reporter channel for the cells; the measurement is the same.

A named segmentation runs only when a pattern method (or tracking) asks for
it, at the cadence it is asked for, exactly like the default one. Its
labels are stored as `labels/<name>` next to the default
`labels/segmentation` (OME-Zarr only; the HDF5 format keeps the default one
and warns once about the rest). To track the nuclei, set
`segmentation = "nuclei"` in `[tracking]`. Both segmentations of a channel
come from the same camera frame, so they line up pixel for pixel.

---

## Changing settings from a pattern method

A method can change the acquisition settings of **its own experiment**
while the run is in progress: a channel's exposure, a config preset, a
device property such as a laser intensity, or the stage position. The
structure of the experiment never changes (same channels, same cadence,
same number of timepoints); only the values inside it do.

```python
context.settings("545")        # what is in force now: exposure_ms, binning,
                               #   config_groups {group: preset}, device_properties {"Dev-Prop": value}
context.settings("stimulation")   # the stimulation channel
context.position()             # {"x", "y", "z", ...} of this experiment

context.set_exposure("545", 50)                          # ms
context.set_config("545", "Channel", "GFP")              # group, preset
context.set_property("farred", "LaserFarRed", "Intensity", 40.0)
context.set_position(z=1234.5)                           # any of x, y, z, pfs_offset
```

A request made while generating the pattern for timepoint `t` applies
from the next timepoint whose acquisitions have not been sent to the
microscope yet, usually `t + 1`, the same delay as the pattern itself. An
invalid request (an unknown channel, a non-positive exposure) is refused
with a warning; the pattern and the other requests still go through. A
preset or property the hardware rejects shows up as an acquisition error.

Every change is recorded twice. `events.parquet` in the experiment
directory has one row per request with the timepoint it was made at, the
timepoint it applied from, the old and the new value, and whether it was
applied or refused. And the frames table gains a column for every device
property or config group a method has changed, holding the value in force
on each frame from the first change on (exposure and position already have
columns), so each frame remains self-describing. See
[Data format](data_format.md).

### Example 4 — a red / far-red switch

Red / far-red optogenetic tools work like a switch: red light turns them
on, far-red turns them off. This method reads a programme string, one
character per timepoint (`1` = red on, `0` = far-red on; the last
character holds), and turns the two lasers on and off through a device
property. Red goes through the DMD as the stimulation channel; far-red is
an ordinary channel of the experiment. The pattern is the whole field, so
the programme alone decides what the cells receive:

```toml
[channels]
group = "Channel"
presets = ["545", "farred"]

    [channels.farred]
    exposure = 500
    save = false

[pattern]
method = "red_farred_switch"
program = "111111000000111111"   # one character per timepoint
farred_channel = "farred"
red_device = "LaserRed"
farred_device = "LaserFarRed"
property = "Intensity"
on = 100
off = 0
```

```{literalinclude} examples/farred_patterns.py
:language: python
:pyobject: RedFarRedSwitch
```

One principle is at work: a setting requested at timepoint `t` applies
from `t + 1`, so the method reads the programme one step ahead. The frames
table then carries `LaserRed-Intensity` on every stimulation frame and
`LaserFarRed-Intensity` on every `farred` frame from the first change on,
and `events.parquet` lists each change with the timepoints involved. The
method needs no image data at all; a closed-loop version would read the
cells and choose the character itself.
