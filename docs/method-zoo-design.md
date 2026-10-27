# Method zoo: coverage, grouping and a thumbnail grid

Status: proposed and implemented 2026-10-26 (PR #43). Harrison's answers
are recorded under each decision in §3; the later ideas in §2 (animation,
filters, directions, retiring duplicates) were taken up too.
§4 lists pattern methods that look missing, §5 what a closed-loop sample
needs.

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

1. **A grid of thumbnails at the top**, one card per registered name in
   family order: thumbnail, family, title, TOML key, what it needs (open
   loop / segmentation / tracks) and whether it changes over time. Each
   card links to its entry.
2. **Every registered name is covered**, and a test keeps it so: a name
   either has its own card, or is listed on the card of the method it
   duplicates.
3. **Entries** give the description, a `[pattern]` table to paste, then
   sample | pattern (what the DMD gets) | overlay (| the target image of
   `centered_image`), and a filmstrip of the overlay at chosen times or,
   for whole-field methods, a timeline of the lit fraction.
4. **The gallery is driven by `known_models`**, not a hand-kept module
   list, so a new built-in without `zoo_meta` fails a test.
5. `rings` registered; `fb_bounce` reversal fixed; the stale docstrings,
   the print and the typo fixed.
6. **Animated thumbnails** for every method that changes over time
   (24 frames over the filmstrip's span or the timeline, animated WebP at
   200 px: about 1.2 MB for the nine, against ~7 MB as GIF).
7. **Filter buttons** above the grid (all, open loop, closed loop,
   changes over time, each family): `documentation/_static/zoo.js`, which
   reads the cards' `data-*` attributes; without JavaScript the grid is
   unfiltered.
8. **Directions fixed**: cells move towards the light (Harrison,
   2026-10-26), so `direction`'s `"u"` lit the wrong side; `"u"` is now
   `(-1, 0)` (up the image) and `"d"` `(1, 0)`. `move_down` was right.
9. **Duplicates retired**: `global_cycle`, `move_down` and `fb_bounce`
   carry `deprecated` (a `PatternMethod` class attribute) naming the
   general method and its arguments; `pyclm check` warns with it. They
   stay registered and behave as before.

Not done:

10. **A time-series sample per closed-loop need** (§5): tracks are faked
    from the segmentation (every object is its own track), so a method
    that depends on history or track identity has no meaningful still.

**Added 2026-10-26 (after PR #43): keyword tables.** Each entry ends
with every keyword the method takes, its default and what it does; the
Methods section opens with the keywords every method takes. The rows come
from `pyclm.check._named_parameters` (the same keywords `pyclm check`
accepts, followed up the MRO; a dispatching base such as `bar` takes the
union of its subclasses') and the descriptions from the `:param name:`
lines of the `__init__` and class docstrings, most derived first.
`tests/test_zoo_gallery.py` fails for a keyword without one, so a new
method documents its keywords in its docstring, once.

## 3. Decisions (answered 2026-10-26)

1. **Grid layout: plain HTML and CSS, no new dependency.** Thumbnails go
   to `_generated/static/zoo/` and are served from `_static/zoo/`.
   *Confirmed.*
2. **Families**: Static shapes (`circle`, `full_on`); Moving stripes
   (`bar`, `bar_bounce`, `sawtooth`, `rotate_bar`, `rings`); Timed and
   blinking (`global_pulse`, `pulse_schedule`, `blinking_bar`); Per-cell
   direction (`move_in`, `move_out`, `rotate_ccw`, `direction`);
   Per-nucleus feedback (`binary_nucleus_clamp`, `centered_image`);
   Embryo axis (`embryo_inner`, `embryo_outer`); Combining methods
   (`split`). *Confirmed.*
3. **Grouped, not given their own card**: `global_cycle`, `move_down`,
   `fb_bounce`. *Confirmed; retire them (§2.9).*
4. **Skipped**: base classes (`NucleusControlMethod`, `PerCellPatternMethod`,
   `PatternAlongAxis`); none is a TOML name. *Confirmed.*
5. **Register the concentric rings** as `rings` (not `wave`).
   *Confirmed with the rename.*
6. **Fix `fb_bounce`.** *Confirmed.*
7. **Methods see camera counts in the zoo.** *Declined: methods keep the
   display-normalised image (0–255), so example thresholds are in those
   units (`clamp_target = 65` lights about half the MCF10A nuclei).*
8. **`centered_image` example** with a target in `zoo_sources/`, shown as
   a panel. *Confirmed, with the Princeton shield as the target*:
   `documentation/zoo_sources/target_shield.tif`, 300 px tall, inverted
   from the file Harrison supplied so the shield is bright (lit) and the
   background dark, and saved uncompressed (`tifffile` reads LZW only
   with `imagecodecs`, which PyCLM does not depend on).

The single-cell direction example keeps the default `gradient = true`
(Harrison).

## 4. Pattern methods that look missing

Ordered by how often a lab would reach for them. "Family" is the zoo
family it would join.

| Proposed name | What it lights | Loop | Family | Why |
|---|---|---|---|---|
| `rectangle` | A rectangle (or half field) given in µm from the field centre | open | Static shapes | The most basic spatial control: stimulated vs unstimulated region in one field. `circle` is the only shape. |
| `image` | A TIF mask projected as is (scaled to the field, optionally thresholded) | open | Static shapes | `centered_image` is closed loop; an arbitrary open-loop mask (a letter, a drawn region) has no method. |
| `gradient` | A static linear (or radial) intensity ramp across the field, dithered | open | Static shapes | Dose–response along one axis in a single field. |
| `spot_array` | A grid of spots, each at its own dose (duty or dithered intensity) | open | Static shapes | Multiplexed dose–response; many conditions per position. |
| `moving_spot` | A spot travelling along a line or a circle at a set speed | open | Moving stripes | Guided migration and wound-edge experiments; a bar is too wide. |
| `random_pulses` | Whole field switched by a seeded random telegraph or white-noise sequence | open | Timed and blinking | System identification of signalling dynamics; the pulse methods are all periodic or listed. |
| `chirp` / `temporal_ramp` | Whole field pulsed with a period (or duty) that changes over the run | open | Timed and blinking | Frequency response of a pathway in one experiment. |
| `all_cells` / `background` | Every segmented object, or everything but them | closed | Per-cell | The simplest closed loop; also the control for per-cell direction (same cells, no polarity). |
| `random_cells` | A seeded random fraction of cells, fixed per track | closed (tracks) | Per-cell | Mosaic stimulation, neighbour effects; needs track identity so the same cells stay lit. |
| `select_cells` | Cells chosen by a property: area, intensity, distance from the centre or the colony edge, top N | closed | Per-cell | Targets a sub-population without writing a method. |
| `colony_edge` | Cells (or a band) within a distance of the tissue boundary | closed | Per-cell | Leader-cell and edge experiments in MDCK sheets. |
| `follow_cell` | One tracked cell (by id or nearest the centre), followed as it moves | closed (tracks) | Per-cell | Single-cell perturbation within a monolayer. |
| `nucleus_proportional` | Each nucleus lit by a duty proportional to its error from a target (P or PI, using history) | closed (history) | Per-nucleus feedback | Bang-bang oscillates; a proportional controller is the standard next step. |
| `population_clamp` | The whole field on or off (or a duty) to hold the mean nuclear intensity at a target | closed | Per-nucleus feedback | Global feedback with a single input; simplest to analyse. |
| `embryo_pole` / `embryo_band` | One pole only, or a band at a given position along the long axis | closed | Embryo axis | `embryo_outer` lights both poles symmetrically; asymmetric stimulation is the common ask. |

**Status (2.0):** built as `rectangle`, `image_mask`, `dose_grid`
(which covers `spot_array`), `chirp`, `all_cells` (`invert` covers
`background`), `random_cells`, `select_cells` (which covers
`colony_edge` with `band_um`), `nucleus_clamp` with `controller = "p"` /
`"pi"` (`nucleus_proportional`) and `population_clamp`. Declined:
`gradient` (as `ramp`), `moving_spot`, `random_pulses`, `follow_cell`,
`embryo_pole` / `embryo_band`. See
[pattern-methods-expansion-design.md](pattern-methods-expansion-design.md).

Graded intensities on a binary DMD need spatial dithering or a duty
cycle across timepoints; one shared helper (`dither(pattern, rng)`)
would serve `gradient`, `spot_array` and `nucleus_proportional`.

## 5. A closed-loop sample: what data it needs

For methods that use history or tracks (`random_cells`, `follow_cell`,
`nucleus_proportional`), the zoo needs a short time series rather than a
still:

- **Raw**: one TIF stack `(T, Y, X)`, uint16, one channel, 10–30
  timepoints at the experiment's interval. A 400 × 400 crop is enough;
  we can crop or bin a larger field.
- **Labels** (optional but preferred): a matching `(T, Y, X)` int32 label
  stack, ideally with track ids already consistent over time (as the
  `tracks` label images in a PyCLM OME-Zarr output are). Without them we
  segment with cellpose and track with the centroid tracker here.
- **Metadata**: pixel size in µm, the interval in seconds, the channel
  and what it shows, display limits (`vmin`, `vmax`).
- A PyCLM output directory (`<experiment>.zarr` plus `tracks.parquet`)
  carries all of this already; `pyclm.io` reads it.

## 6. How it was checked

- `uv run --group test pytest tests/test_zoo_gallery.py`: coverage of
  `known_models`, and the gallery built into a temporary folder with
  one thumbnail per card.
- A local `uv run --group docs sphinx-build -b html documentation <out>`
  and a look at the page.
