# Method Zoo

Visual gallery of all built-in pattern methods.  Each card shows an example
raw image alongside the illumination pattern the method would generate with
the parameters listed.  Gallery images are rendered automatically at docs
build time from sample images in `documentation/zoo_sources/`.

## Stimulation / Pattern Methods

```{include} _zoo_gallery.md
```

> **Adding a new method to the gallery** — attach a `zoo_meta` class variable
> to your `PatternMethod` subclass (see `src/pyclm/core/patterns/zoo.py` for
> the `ZooMeta` dataclass) and add the corresponding source image to
> `documentation/zoo_sources/<source>.tif`.  A `<source>_seg.tif` label image
> is optional; synthetic blobs are used when absent.
