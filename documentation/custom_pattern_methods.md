# Custom Pattern Methods

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

The subclass needs to implement at least two methods: `__init__` and `generate`. 
First, the `__init__` method, which is standard for python: it is called when an instance of the custom pattern class is created.
PatternMethods objects are generated for each position, and are instantiated as the experiment is starting.
The `__init__` method defines keyword arguments that the user can pass in via `.toml` files.

The final function of the `__init__` method is to request that the ``generate`` method be supplied with imaging data 
and/or segmented images. This is done by calling ``self.add_requirement``.
```python
class MyCustomPattern(PatternMethod):
    ...
    # the below __init__ function definition defines two 
    # keyword arguments that can be supplied in the .toml
    def __init__(self, keyword_a="default_value_a", keyword_b=42, **kwargs):
        super().__init__(**kwargs)  # boilerplate, always needs to be present

        # store keywords
        self.attribute_a = keyword_a
        self.attribute_b = keyword_b

        # request raw image data from e.g. RFP channel
        self.add_requirement("RFP", raw=True, seg=False)

        # request the segmentation of the GFP channel
        self.add_requirement("GFP", raw=False, seg=True)
```

The second method is the `generate` function, which actually produces the pattern to be supplied to the DMD. 
`generate` takes in a `PatternContext` object (described below) and returns a pattern. The pattern should be the same 
size and shape as the data coming out of the camera. PyCLM will automatically translate this pattern to the DMD.

```python
from pyclm import PatternContext, PatternMethod

class MyCustomPattern(PatternMethod):
    ...
    def generate(self, context: PatternContext):
        
        # you may choose to unpack data from the context
        time = context.time  # current time of the experiment in seconds since start

        # unpack data requested in __init__ using add_requirement()
        raw = context.raw("RFP")  # the most recent data in the RFP channel
        seg = context.segmentation("GFP")

        # gets the x coordinates and y coordinates of the camera in microns
        xx, yy = self.get_um_meshgrid()

        # gets the center of the image in microns
        xc, yc = self.center_um()

        # you must return a pattern the same size as the input image
        # raw, seg, xx, and yy all have the same shape as the camera output
        # e.g. illuminate the right half of the image a percentage specified by the user:
        pattern = (self.keyword_b / 100) * (xx > xc)

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
    """Illuminates a ring centred on the FOV."""

    name = "annulus"

    def __init__(self, inner_radius_um=30, outer_radius_um=60, **kwargs):
        super().__init__(**kwargs)
        # No add_requirement calls — this pattern needs no image data.
        self.r_inner = inner_radius_um
        self.r_outer = outer_radius_um

    def generate(self, context) -> np.ndarray:
        xx, yy = self.get_um_meshgrid()
        cy, cx = self.center_um()

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
from pyclm.core.patterns.pattern import PatternMethod


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
| `self.center_um()` | returns `(cy, cx)` centre of the FOV in µm |
| `self.add_requirement(channel_name, raw, seg)` | declare that `generate` needs raw/seg data for a channel |
| `self.request_stim(raw, seg)` | same, but for the stimulation-output channel |

The `generate` method receives a `PatternContext` and must return a `float` array with values in `[0, 1]` and shape matching `self.pattern_shape`.

```python
context.raw(channel_name)           # np.ndarray – raw fluorescence image
context.segmentation(channel_name)  # np.ndarray – labelled segmentation mask
context.stim_raw()                  # raw image of the stimulation channel
context.time                        # elapsed experiment time in seconds
```

---