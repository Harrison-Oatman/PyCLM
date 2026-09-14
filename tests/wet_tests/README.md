# PyCLM wet tests

The pre-release checks that need a microscope (`docs/release-wet-tests.md`
in the repository lists them and what each guards). This directory is
self-contained: copy it anywhere, point `pyproject.toml` at the PyCLM
checkout you are releasing, and run the items in order.

## Setup, once

1. Copy this directory next to (not inside) the PyCLM checkout, or
   anywhere; in `pyproject.toml` set the `path` of `pyclm` under
   `[tool.uv.sources]` to that checkout. Then

   ```
   uv sync
   uv run wet list
   ```

2. Fill in `config_template.toml`: the MicroManager `.cfg`, the affine from
   the Projector calibration, the DMD shape, `position_mover = "pfs"` on a
   Nikon. Leave `camera_roi` empty for now; item 01 tells you what to put.
   The runner copies this file into every experiment directory as
   `pyclm_config.toml` (applying the directory's `overrides.json`, if any),
   so you edit it once.

3. Read `experiments/<item>/guide.md` before each item: it says what must
   be generated live (the position list from MicroManager, the sample, the
   camera ROI) and what "pass" means.

## Running an item

```
uv run wet prepare 02      # copy the config, apply overrides, run pyclm check
uv run wet run 02          # prepare + pyclm run + verify -> results/02_open_loop_bar.md
uv run wet verify 02       # re-run the verification on an existing result
uv run wet commands 08     # item 08: send the command sequence to a run in progress
uv run wet report          # collate results/*.md into results/REPORT.md
uv run wet run 02 --dry    # rehearse an item on the virtual microscope (needs TIFs, see the guide)
```

`run` registers the custom methods the items use (`wet/methods.py`) and
passes them to `run_pyclm`, exactly as a user script would. Every run's
outputs stay in its experiment directory; `wet clean <item>` removes them
so the item can run again (the run refuses to overwrite).

## What to keep

`results/REPORT.md` plus, for each item, the experiment directory's
`log.log`, `status.json`, `events.parquet` and `frames.parquet`. The report
names the commit of the PyCLM checkout it ran against.

## Items

| Item | What | Needs live |
|---|---|---|
| 01_environment | the core loads, a snap through PyCLM | the microscope on, `config_template.toml` filled |
| 02_open_loop_bar | the reference run, PFS per position | a two-position list, any sample |
| 03_hdf5 | the same in HDF5 format 1 | the list of 02 |
| 04_closed_loop_cellpose | cellpose + a per-cell method | live cells, a one-position list |
| 05_tracking | 04 with tracking | the list of 04 |
| 06_ktr | two segmentations of one channel | KTR cells, a one-position list |
| 07_settings_ramp | a method changing its own settings | the list of 02 |
| 08_commands | commands from outside during a run | the list of 02 |
| 09_farred_switch | the red / far-red programme | a red + far-red channel pair, one position |
| 10_skipping | blank patterns skip the position | the list of 02 |
| 11_camera_roi | the DMD footprint as the ROI | one position, a fluorescent slide |
| 12_grid | a 2 x 2 grid | Create Grid on a structured slide |
| 13_overnight | 8 h of the lab standard | cells, the lab's list |
| 14_error_recovery | a provoked hardware error | the list of 02 |
| 15_restart_safety | the overwrite refusal | the outputs of 02 |
| 16_viewer_export | the viewer and the export | the outputs of 02, 04, 12 |
