# Method Zoo

Every built-in pattern method, run on a sample image. The cards below show
the pattern (cyan) over the image, animated for methods that change over
time; the buttons above them filter by family or by open / closed loop.
Each card links to the method's entry: a `[pattern]` table that reproduces
it, the image, the pattern the DMD receives, and how it changes over time.
The images are rendered at docs build time from the samples in
`documentation/zoo_sources/`, scaled to 0–255 before the method sees them,
so intensity thresholds in the examples are in those units.

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
