# Method zoo: coverage, grouping and a thumbnail grid

Status: proposed 2026-10-26, implemented on branch `claude/project-thread-lweink`
with every decision as recommended (draft PR). Each decision is small and
reversible; say which to change.

## 1. Where the zoo stood

The method zoo (`documentation/method_zoo.md`) is built at docs build time by
`documentation/ext/zoo_gallery.py`: every `PatternMethod` subclass with a
`zoo_meta` class variable (`src/pyclm/core/patterns/zoo.py:ZooMeta`) is run
once on a sample image from `documentation/zoo_sources/` and rendered as a
400 × 400 cyan overlay. The page is one long column: heading, image,
sentence, TOML key, parameter table, rule, repeated.

Measured on `main` at 0409a8e (a local `sphinx-build`, contact sheet of the
14 generated PNGs):

**Coverage.** 21 names are registered in `known_models`
(`src/pyclm/core/patterns/__init__.py`). 12 have a card. 9 do not:
`split`, `bar_bounce`, `sawtooth`, `global_pulse`, `pulse_schedule`,
`centered_image`, `move_down`, `fb_bounce`, `direction`. Two cards are for
things a TOML cannot name: "Generic Nucleus Control" (the
`NucleusControlMethod` base class) and "Radial Wave" (`WavePattern`, whose
`name = "wave"` is not in `known_models`, so `method = "wave"` fails
`pyclm check`).

**Redundancy.** Several registered names are other methods with fixed
arguments:

| Name | Same pattern as |
|---|---|
| `global_cycle` | `global_pulse` with `period_min = 2 × period_m`, `on_min = period_m` (checked: `(t // P) % 2 == 0` ⇔ `t % 2P < P`) |
| `move_down` | `direction` with `sequence = "u"`, `voronoi = false`, `gradient = false` (both use the vector `(1, 0)`) |
| `fb_bounce` | `direction` with `sequence = "ud"`, `period_h = t_loop / 120`, `voronoi = false`, `gradient = false`, once the bug below is fixed |

`global_cycle` also requires a `channel` it never reads (it subclasses
`NucleusControlMethod`), so an open-loop pattern needs a segmented channel.

**Examples that mislead.** The sample image is normalised to 0–255 before
the method sees it, so thresholds in examples are in display units, not
camera counts: `binary_nucleus_clamp` uses `clamp_target = 75`, which
lights every nucleus and makes the card identical to the base class's.
`global_cycle` looks identical to `full_on`. Time-varying methods are shown
at one instant, so `bar`, `bar_bounce` and `sawtooth` would be
indistinguishable, and a pulse train is a full cyan square.

**Small faults found on the way.** `BounceModel` (`fb_bounce`,
`src/pyclm/core/patterns/fbc_cell_movement.py:BounceModel.generate`) sets
`self.down = False` at the first half-loop and never sets it back, so after
`t_loop / 2` it points one way for the rest of the run. The zoo's
`make_zoo_subclass` returns `center_um()` as `(y, x)` where the base class
returns `(x, y)` (harmless on the square samples). `_render_overlay` prints
every image shape. The module docstring says the PNGs land in `_static/zoo/`
and that a missing `_seg.tif` falls back to synthetic blobs; neither is
true. "centerd" in the circle description.

## 2. Improvements, in order of value

Done in this PR (decisions in §3):

1. **A grid of thumbnails at the top**, grouped by family, one card per
   registered name: thumbnail, title, TOML key, what it needs (open loop /
   segmentation / tracks) and whether it changes over time. Each card links
   to its detail entry below.
2. **Every registered name is covered**, and a test keeps it so: a name
   either has its own card, or is listed on the card of the method it
   duplicates ("also: `global_cycle`, same as … with …").
3. **Detail entries** show raw | pattern (what the DMD gets) | overlay, and
   for time-varying methods a filmstrip of the overlay at chosen times, or
   for whole-field methods a timeline of the lit fraction. The parameter
   table becomes a `[pattern]` TOML block to paste.
4. **Methods see camera counts**: the sample image is passed in its own
   units and normalised only for display, so example thresholds are
   realistic (`clamp_target = 1300` lights about half the MCF10A nuclei).
5. **The gallery is driven by `known_models`**, not a hand-kept module
   list, so a new built-in without `zoo_meta` fails a test instead of
   silently missing.
6. `wave` registered; `fb_bounce` reversal fixed; the stale docstrings,
   the print and the typo fixed.

Not done, worth considering later:

7. **Animated thumbnails** (a short GIF per time-varying method) instead
   of or as well as the filmstrip. Cheap to render; whether a page of
   moving cards helps or distracts is a taste call.
8. **Filtering** the grid by family or by "open loop / closed loop"
   (needs a little JavaScript in `_static/`).
9. **Directions are named inconsistently**: `direction`'s `"u"` and
   `move_down` use the same vector `(1, 0)`, which lights the half of each
   cell towards +y (down on screen). Which name is right depends on which
   way the cells respond; a fix is a rename with an alias.
10. **Retire the duplicates** (`global_cycle`, `move_down`, `fb_bounce`)
    as deprecated aliases that build the general method, after a release
    in which `pyclm check` warns. They stay registered here.
11. **A sample per closed-loop need**: tracks are faked from the
    segmentation (every object is its own track), so a method that
    depends on history or track identity has no meaningful still.

## 3. Decisions to confirm

1. **Grid layout: plain HTML and CSS, no new dependency.** Thumbnails go
   to `_generated/static/zoo/` and are served from `_static/zoo/`; the grid
   is a raw-HTML block in the generated fragment, styled in
   `_static/pyclm.css`. Alternative: `sphinx-design` cards (one more docs
   dependency, theme-aware). *Recommended: plain HTML.*
2. **Families**, in this order: Static shapes (`circle`, `full_on`);
   Moving stripes (`bar`, `bar_bounce`, `sawtooth`, `rotate_bar`, `wave`);
   Timed whole-field and blinking (`global_pulse`, `pulse_schedule`,
   `blinking_bar`); Per-cell direction (`move_in`, `move_out`,
   `rotate_ccw`, `direction`); Per-nucleus feedback
   (`binary_nucleus_clamp`, `centered_image`); Embryo axis
   (`embryo_inner`, `embryo_outer`); Combining methods (`split`).
3. **Grouped, not given their own card**: `global_cycle` (under
   `global_pulse`), `move_down` and `fb_bounce` (under `direction`). Each
   is listed with its equivalent arguments. *Recommended.*
4. **Skipped**: the `NucleusControlMethod` base class (not a TOML name).
5. **Register `wave`** (as `WavePatternBase`, so `wave_speed = 0` gives the
   stationary rings like `bar` does). It already had a card and works.
6. **Fix `fb_bounce`** so it reverses every half-loop, as its docstring
   and `bar_bounce` do. Behaviour change for runs longer than one loop.
7. **Methods get camera counts in the zoo** (§2.4).
8. **`centered_image` example** uses a new 200 × 200 radial target,
   `documentation/zoo_sources/target_radial.tif`; a string argument that
   names a file in `zoo_sources/` is resolved to it.

## 4. How it was checked

- `uv run --group test pytest tests/test_zoo_gallery.py`: coverage of
  `known_models`, and the gallery built into a temporary folder with
  one thumbnail per card.
- A local `uv run --group docs sphinx-build -b html documentation <out>`
  and a look at the page.
