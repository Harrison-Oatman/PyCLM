# Method Zoo

Every built-in pattern method, run on a sample image. The cards below show
the pattern (cyan) over the image; each links to the method's entry, with
the image, the pattern the DMD receives, how it changes over time, and a
`[pattern]` table that reproduces it. The images are rendered at docs build
time from the samples in `documentation/zoo_sources/`, and the methods see
them in camera counts, so thresholds in the examples are realistic.

Two methods can share one field, each on its own half, with `method =
"split"` (under Combining methods below); see the "Combining methods"
section of [Writing custom pattern methods](custom_pattern_methods.md).

```{include} _zoo_gallery.md
```

> **Adding a method to the gallery** — attach a `zoo_meta` class variable
> to your `PatternMethod` subclass (see `src/pyclm/core/patterns/zoo.py` for
> the `ZooMeta` dataclass): a sample image (`source`), the example's
> arguments, a family (`group`), and `times` or `timeline` if the pattern
> changes over time. A built-in that is another method with fixed
> arguments names that method with `same_as` and is listed on its card.
> `tests/test_zoo_gallery.py` fails for a registered built-in without
> `zoo_meta`.
