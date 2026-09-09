# Stage 5b: the viewer, a control window, and commands during a run — options and a recommendation

Date: 2026-09-08. Status: **implemented** 2026-09-09 on `core-refactor`
with every decision in §8 taken as recommended. Code: `gui/widgets.py`
(status line, minimap, `RunOverview`, docked in the napari viewer),
`commands.py` and `Manager.poll_commands` / `apply_command` (a `commands/`
directory polled at the boundary; pause shifts the clock, `stop_run`,
`stop_experiment` through `AcquisitionPlan.stop_experiment`, `set_*`
through `apply_settings` with `source = "command"`, `set_pattern` through
`PatternMethod.update`), `gui/control.py` (`pyclm control`: run panel with
a `pyclm run` subprocess, positions panel on `MMCoreStage` /
`SimulatedStage` with pymmcore-widgets for the real stage,
`directories.write_position_list`), `gui/forms.py` (schema-generated forms,
`tomlkit` merge that changes only what changed). Dependencies: pymmcore-plus
0.18.1, pymmcore ≥ 12.5, useq-schema ≥ 0.9.2, napari ≥ 0.9,
pymmcore-widgets ≥ 0.12.1, tomlkit. Tests: `tests/test_gui_widgets.py`,
`tests/test_commands.py`, `tests/test_control.py` (offscreen Qt); the
microscope validation is in [testing-guide-stage5b.md](testing-guide-stage5b.md).
One deviation from §5: the Micro-Manager demo configuration is not
installed on the development PC, so the pymmcore-widgets pane was written
but only exercised by import; the positions logic is tested on the
simulated stage.

The question this stage answers first is which tools to stand on. The
current viewer (`gui/gui_controller.py`, ~250 lines of napari) is a good
read-only viewer of a run, live or finished, and the user's instinct is to
keep it that way and put interactivity elsewhere, on existing tools from
the smart-microscopy field, with new dependencies allowed if they look
stable for one to two years. This document records what those tools are
today (verified, §1), the requirements as stated (§2), three ways to cut
it (§3), a recommendation (§4), and what the recommended cut contains
(§5–§7).

---

## 1. What exists today (verified 2026-09-08)

| Package | Installed | Latest | Released | Status | Notes |
|---|---|---|---|---|---|
| napari | 0.7.0 | 0.9.1 | 2026 | Beta, Python ≥ 3.11 | Image, labels, shapes, points layers; plugin ecosystem (`napari-ome-zarr`); embeddable as a Qt widget; heavy to import. |
| ndv | – | 0.5.0 | 2026-04-08 | Alpha (classifier "3 - Alpha") | "Simple, fast-loading, asynchronous, n-dimensional array viewer, with minimal dependencies" (cmap, ihist, numpy, psygnal, pydantic). Qt / Jupyter / wx front ends, vispy or pygfx. Composites channels with colormaps; a writable `data` property and `current_index.update()` for live data; a rectangular ROI. **No label, shapes or points layers.** 0.4 → 0.5 replaced its internal display model. |
| pymmcore-widgets | – | 0.12.1 | 2026-03-27 | active (0.11 Feb, 0.12 Mar 2026) | `StageWidget`, `PositionTable`, `GridPlanWidget`, `ChannelTable`, `ExposureWidget`, `ImagePreview`, `SnapButton` / `LiveButton`, `PropertyBrowser`, `MDAWidget` (produces useq); qtpy, PySide6 ≥ 6.9 supported; **requires pymmcore-plus ≥ 0.15.4**. |
| pymmcore-gui | – | 0.0.1rc0 | 2025-08-27 | Alpha, one release candidate | A whole MMStudio replacement on pymmcore-widgets + ndv + a QMainWindow with docks and an IPython console. Pins **PyQt6 == 6.9.1** (PyCLM uses PySide6) and **zarr < 3** (PyCLM uses zarr 3): cannot be installed alongside PyCLM. Its architecture is a template; the package is not a dependency. |
| pymmcore-plus | 0.13.7 (Apr 2025) | 0.18.1 | 2026-04-02 | active (monthly) | Release notes 0.14–0.18 list additions (MDA sinks, UniMMCore, ROI on setup events) and no removals of the CMMCorePlus methods PyCLM uses (`setConfig`, `setProperty`, `snapImage`, `getImage`, `setSLMImage`, `getROI`, `waitForSystem`, `loadSystemConfiguration`, `setFocusDevice`, `getConfigGroupObject`, …). Requires pymmcore ≥ 11.10 and useq-schema ≥ 0.9. |
| useq-schema | 0.8.1 | 0.9.2 | 2026 | active | 0.9 added ROI on `MDAEvent`, setup parameters on `MDASequence`, plate metadata on positions. |

**Compatibility, measured.** `uv pip install --dry-run "ndv[qt,vispy]"
pymmcore-widgets` against the project's pins resolves only by upgrading
pymmcore-plus 0.13.7 → 0.18.1 and pymmcore 11.2 → 12.5. In a scratch
environment with pymmcore-plus 0.18.1, pymmcore 12.5.0.75.0, useq-schema
0.9.2, pymmcore-widgets 0.12.1, ndv 0.5.0 and napari 0.9.1, **the whole
PyCLM suite passes (215 tests, 127 s)**: the plan on useq 0.9, both
storage formats, the router, the dry runs on the simulated core. What the
suite cannot cover is `RealMicroscopeCore` on the hardware; the methods it
calls are the basic CMMCore API and the release notes name no changes to
them, but the first run after the upgrade must be a short known
experiment on the microscope (§7).

---

## 2. Requirements, as stated

- A status line that updates (no lateness plot); a **minimap** in the
  viewer showing where each position is.
- Operator commands during a run: pause / resume, stop, change an
  exposure, a laser property, a position, a pattern parameter.
- Positions with meaning (move, snap, name, link to an experiment, preview
  there) and forms from the schema. No parameter-tuning sliders.
- One window is welcome **if** it cannot jeopardise a run: the viewer is a
  convenience and a crash on its side must not affect results.
- No human-in-the-loop pattern methods; no phone page.
- New dependencies allowed if stable for ~1–2 years.

---

## 3. Three ways to cut it

### A. napari only: extend the current viewer with docks

Add the minimap and status as napari dock widgets, add control widgets
(pymmcore-widgets) as further docks in the same napari window, commands
through the run's boundary.

*For:* one window, no new viewer. *Against:* the control side (stage,
snap, positions) needs the microscope core, which napari's process does
not own during a run, so the docks would be half-functional half the
time; napari's start-up cost and plugin machinery come along for the ride
in what is meant to be a light control app; a napari crash takes the
controls with it.

### B. A new unified app on ndv + pymmcore-widgets, modelled on pymmcore-gui

Replace napari with an ndv canvas inside a QMainWindow with docked
pymmcore-widgets, as pymmcore-gui does.

*For:* the lightest viewer; the same stack the pymmcore ecosystem is
converging on; pymmcore-gui shows the layout works. *Against:* ndv is alpha
(0.5.0, internals rewritten between 0.4 and 0.5) and has **no label,
shapes or points layers**, so segmentations, tracks and the DMD overlay,
which the viewer shows today through napari's labels layer, would have to
be composited into RGB by hand and lose the ability to click a cell and
read its id; the OME-Zarr reading that napari's plugin gives for free would
be ours to write; pymmcore-gui itself cannot be a dependency (PyQt6 and
zarr < 3 pins). This is the right direction in two years if ndv grows
layers; it is a rewrite with a feature loss today.

### C. Keep napari as the viewer; a new control window on pymmcore-widgets, in its own process; shared plain-Qt widgets

Three processes, joined only by files in the experiment directory:

- **The run** (`pyclm run`): owns the microscope core; no Qt. Reads
  `commands/` at the timepoint boundary Stage 4 established; writes
  `status.json`, `events.parquet`, the data.
- **The viewer** (`pyclm gui`, napari as today): read-only; gains a status
  line that updates and a minimap dock; can be closed, crashed or reopened
  at any time.
- **The control window** (`pyclm control`, new, PySide6 + pymmcore-widgets):
  owns the core **only while no run is active** (positions window: stage,
  snap, live, position table with an experiment column, preview at the
  current position); forms for the three files generated from the schema;
  a run panel that shows the check report, starts `pyclm run` as a
  subprocess, shows status, and writes commands. If it crashes during a
  run, the run does not notice.

The minimap and the status line are plain `QWidget`s that both the viewer
(as a napari dock widget) and the control window use, so they are written
once and do not care which host shows them. If ndv becomes the viewer one
day, the same widgets dock beside it.

---

## 4. Recommendation: C

It meets the stability requirement by construction (the run has no GUI
in its process and nothing waits on a GUI), keeps napari's mature layers
for what the viewer is good at, takes the one part of the pymmcore
ecosystem that is both stable and exactly what "positions with meaning"
needs (pymmcore-widgets, on a monthly release train since 2024), and
leaves the viewer choice open at low cost, since the viewer is small and
the new widgets are host-independent. It costs the pymmcore-plus upgrade,
which the suite has already passed and one microscope run will confirm.

---

## 5. What the control window contains

**Positions window** (pymmcore-widgets): `StageWidget` (XY and Z),
`SnapButton` / `LiveButton` with `ImagePreview` and `ExposureWidget` on the
channel presets of the loaded experiment files, a `PositionTable` with one
added column, *experiment*, offering the TOML stems in the directory, and
three buttons: *add current position* (label `<experiment>.<n>`), *preview
here* (runs `pyclm preview --snap` for that experiment on the current
field, shows the overlay), *save* (writes `PositionList.pos` in
MicroManager's format, which `directories.positions_from_pos` already
reads). The core is loaded from `pyclm_config.toml` when the window opens
and released when a run starts.

**Forms**: one tab per file, generated from the schema (`schema.field_table`
gives type, default, limit and description per key; a `MethodTable` shows
`method` as a dropdown of registered names and the method's constructor
arguments below it, from the same signature `pyclm check` reads). Written
back with `tomlkit` so comments and layout survive; validated live with
the schema's own messages. Named tables (`[channels.<preset>]`,
`[segmentation.<name>]`) are added and removed with buttons.

**Run panel**: the check report (rerun on save), *rehearse* (`--dry`),
*start*, *pause / resume*, *stop run*, per experiment *stop*, and the
setting controls (exposure, a device property, a position nudge, a pattern
parameter) that write commands. Status from `status.json` every second.

---

## 6. Commands during a run

**Transport.** A `commands/` directory in the experiment directory. Each
command is one small JSON file written atomically (temp name, then
rename); the Manager lists the directory at the timepoint boundary,
applies each in name order, records each in `events.parquet`
(`kind = command`, the command text, applied / refused with the reason,
`t_requested` = the timepoint it was found at, `t_applied`), and moves it
to `commands/done/`. Anything can write a command: the control window, a
script, a person with a text editor.

**Command set** (a pydantic model, `pyclm.commands`), all applied at the
boundary and none changing the structure of the plan:

| Command | Fields | Effect |
|---|---|---|
| `pause` | – | the Manager holds before the next burst; the schedule's clock is shifted by the paused time so the interval is preserved on resume |
| `resume` | – | continue |
| `stop_run` | – | end after the current timepoint (graceful drain, outputs finalised) |
| `stop_experiment` | `experiment` | that experiment's `t_stop` becomes its current relative timepoint; others continue (cadence groups are sized from the plan and skipped frames cost nothing) |
| `set_exposure` | `experiment`, `channel`, `ms` | as a pattern method's `set_exposure` |
| `set_config` | `experiment`, `channel`, `group`, `preset` | as `set_config` |
| `set_property` | `experiment`, `channel`, `device`, `property`, `value` | as `set_property` |
| `set_position` | `experiment`, any of `x`, `y`, `z`, `pfs_offset` | as `set_position` |
| `set_pattern` | `experiment`, `parameters` | calls the method's new `update(**parameters)` hook in the pattern process (default: sets attributes that exist, refuses others); recorded like a setting |

The setting commands reuse `Manager.apply_settings` with the command as
the requester, so the frames-table override columns and the events rows
come for free. `pause` and `stop_run` are the only new Manager behaviour.

**Status.** `status.json` gains `paused`, `stopping` and the count of
pending commands, so the viewer's line and the control window agree.

---

## 7. The upgrade and its validation

Pins: `pymmcore-plus == 0.18.1`, `pymmcore >= 12.5.0.75.0`,
`useq-schema >= 0.9.2`, `napari >= 0.9`; new: `pymmcore-widgets >= 0.12.1`,
`tomlkit`. ndv is not adopted. Validation: the suite (already green on the
stack), `pyclm check --dry` on the lab's directories, then on the
microscope `pyclm preview --snap` on a known experiment followed by a short
known run, before any other Stage 5b work lands on the scope.

---

## 8. Decisions to confirm

1. **Viewer stack.** *Recommended:* C, napari stays the viewer, a separate
   control window on pymmcore-widgets, minimap and status as shared plain
   Qt widgets. *Alternative:* B, a unified ndv-based app now, accepting the
   loss of label layers.
2. **Upgrade** pymmcore-plus to 0.18.1 (with pymmcore 12.5, useq 0.9.2,
   napari 0.9) and add pymmcore-widgets and tomlkit, validated as in §7.
   *Alternative:* stay pinned and write the positions window on raw
   CMMCorePlus calls (more code, no upgrade).
3. **Command transport**: one JSON file per command in `commands/`,
   applied at the boundary, moved to `commands/done/` after recording.
   *Alternative:* a socket (needs a server in the run process).
4. **Forms write files with `tomlkit`** so comments survive. *Alternative:*
   plain rewriting, which loses comments.
5. **Order of work**: minimap + updating status in the viewer (small,
   immediately useful) → the upgrade → commands and the run panel →
   positions window → forms.
