---
name: pyclm-write-pattern-method
description: Write, test and run a custom PyCLM PatternMethod (or segmentation/tracking method) for a stimulation rule no built-in method covers. Use when a researcher describes a feedback rule or light pattern in words ("light cells whose nuclear/cytosolic ratio is below X", "a spot that follows each cell") and wants it as code, or asks how to extend PyCLM.
---

# Write a PyCLM pattern method

First confirm that no built-in method, or a `split` of two built-in
methods, already does it (`documentation/method_zoo.md`). If one does, use
`pyclm-design-experiment` instead.

**Where the documentation is.** In a clone of the repository, the files
named below are under `documentation/`, `src/` and `tests/`. After a plain pip
install, read the same pages on https://pyclm.readthedocs.io (for example
`documentation/method_zoo.md` is
https://pyclm.readthedocs.io/en/latest/method_zoo.html), and the source on
https://github.com/Harrison-Oatman/PyCLM, or in the installed package
(`python -c "import pyclm; print(pyclm.__file__)"`).

## The contract

- Subclass `pyclm.PatternMethod` (`from pyclm import PatternMethod`). Set
  the class attribute `name`; it is the `method` value in the TOML. Write
  `__init__(self, <tunable args with defaults>, **kwargs)`: call
  `super().__init__(**kwargs)` first, store the arguments, and declare
  what `generate` needs with
  `self.add_requirement(channel, raw=, seg=, tracks=, history=)`.
  `channel` is a preset name from the experiment's `[channels]`.
  `seg=True` asks for the default `[segmentation]`
  table, `seg="name"` for a `[segmentation.name]` table. `history=n`
  keeps the last n deliveries for `context.history`.
- `generate(self, context) -> np.ndarray`: a float array in [0, 1] of
  shape `self.pattern_shape`, in **camera** coordinates (the binned
  camera frame the method's channel delivers). PyCLM maps it to the DMD.
  Never apply the affine yourself. PyCLM sends the DMD 0 to 255; whether
  values between 0 and 1 give intermediate light depends on the device
  and its adapter (many DMDs are binary), so treat the pattern as on/off
  unless the researcher has checked gray levels on their rig.
- Every argument of `__init__` is a TOML key under `[pattern]`. Use
  physical units in the names (`radius_um`, `period_min`), not pixels.
- Derive anything that depends on the field from `self.get_um_meshgrid()`,
  `self.center_um()` and `self.pixel_size_um`. Do not ask the researcher
  for measured constants such as the field size in pixels; binning and
  objective changes would break them.
- Read only what was declared: `context.raw(ch)`,
  `context.segmentation(ch, name)`, `context.regions(ch)`,
  `context.tracks(ch)`, `context.history(...)`, `context.last_pattern()`,
  `context.t`, `context.time`. Asking for an undeclared channel fails.
- For per-cell rules, use the measurement toolbox (`from pyclm import
  Regions, PerTrack, nuclear_cytosolic_ratio`): `context.regions(ch)` and
  `context.tracks(ch)` return `Regions`, whose `measure(image, stat)` gives
  one value per cell and `paint(values)` turns per-cell values into a
  pattern; `PerTrack` keeps a value per track id across timepoints
  (`documentation/custom_pattern_methods.md`, "The measurement toolbox").
  Prefer these to hand-written regionprops loops.
- Return zeros to mean "no light this timepoint". For a stimulation-only
  visit, PyCLM then skips the stage move and the exposure.
- Settings changes such as exposure and presets go through
  `context.set_exposure/set_config/set_property` and apply from the next
  timepoint.

Read `documentation/custom_pattern_methods.md` for the full API and four
worked examples (static pattern, feedback controller, two segmentations
of one channel, a red / far-red switch that changes settings). Copy the closest example rather than starting from a
blank file.

## Workflow

1. **Write the method** in a file next to `pyclm_config.toml`, for
   example `my_patterns.py`, with a docstring that states the rule in the
   researcher's words, and the units of every argument. Set `name` on the class; it must not be a
   built-in method's name.
2. **List the file** in `pyclm_config.toml`: `methods = ["my_patterns.py"]`
   (a top-level key, above `[output]`). `pyclm check` prints
   "pattern method 'my_rule' from <the file>" when it loaded, or the
   import error when it did not. Python files listed in `methods` are
   run on import, so list only files the lab wrote or trusts. A lab package can instead register its
   methods with a `pyclm.methods` entry point.
3. **Unit-test it without a microscope.** Construct the method, call
   `configure_system` as `tests/test_composite.py` does, and feed it a
   synthetic label image as `tests/test_preview.py` does (both in a
   clone, or on GitHub). Assert on the pattern: which cells are lit, and
   that values are in [0, 1]. Run with `pytest`.
4. **Preview it on a real frame**:
   `pyclm preview <dir> <condition> --image frame.tif`. Show the
   researcher the written pattern.
5. **Check and rehearse**: `pyclm check <dir>`, then
   `pyclm run <dir> --dry --gui`.
6. **Inspect the dry-run output**:
   `pyclm.io.open(dir / "<label>.zarr").camera_pattern_at(t)` against
   the frames. Remember that the pattern from t is shown at t+1.

## Custom segmentation or tracking

These follow the same pattern. Subclass `SegmentationMethod` and
implement `segment(self, data) -> np.ndarray`, returning an integer label
image (0 = background) the height and width of `data`. `data` is 2D,
or `(height, width, channel)` when the table lists `channels`; a method
that takes several channels sets `max_channels` (default 1, Cellpose 3).
Select it in the TOML with
`[segmentation] method = "<name>"`; extra keys in that table are its
arguments. Subclass `TrackingMethod`, where
`track(labels, t, pixel_size_um)` returns the relabelled image and rows.
List them in `methods` the same way. In Python,
`segmentation_methods=` / `tracking_methods=` on `run_pyclm` and
`check_directory` also work and take precedence.

## Before declaring done

The method must have:

- a unit test;
- a preview the researcher has looked at;
- a clean `check`;
- a dry run whose stored patterns match the rule.

State which of these were not done.
