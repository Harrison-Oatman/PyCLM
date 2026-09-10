# docs/ — engineering notes for PyCLM

This folder holds **internal engineering notes** about the PyCLM codebase: how the
core pipeline is wired, what is known to be fragile, and what the plan is for
changing it. It is written for developers and for AI agents working on the repo.

It is deliberately separate from `documentation/`, which is the Sphinx source for
the **user-facing** docs published on readthedocs. Nothing in this folder is
built or published.

| File | What it is | Update when |
|---|---|---|
| [architecture-notes.md](architecture-notes.md) | Factual map of the runtime: processes, queues, message and data types, timing model, shutdown protocol, HDF5 layout, coordinate frames, config parsing. No opinions. | Any change to `src/pyclm/core/`, `controller.py`, `directories.py`, or a storage layout. |
| [known-issues.md](known-issues.md) | Concrete bugs, hazards, and smells found in the code, each with a file reference and a suggested fix. Numbers are stable (the assessment cites them). | An issue is fixed (collapse it to a one-line **Fixed** note, keep the number) or a new one is found (append). |
| [assessment-2026-09.md](assessment-2026-09.md) | The September 2026 architecture assessment: what works, what limits the five planned features (tracking, runtime edits, z-stacks, grids, interactive setup), and a staged refactor roadmap. | Roadmap decisions change. Keep as a dated record; write a new dated file for a later assessment rather than rewriting history. |
| [stage1-plan-design.md](stage1-plan-design.md) | Stage 1 evaluation: can useq's `MDASequence` be PyCLM's acquisition plan (yes, with a thin wrapper), what was verified, the proposed `AcquisitionPlan` design, and decisions to confirm. | Stage 1 decisions are made or the plan class lands. |
| [stage2-storage-options.md](stage2-storage-options.md) | Stage 2 options for data storage rated against a usability bar (Fiji, napari, plain Python, durability, live append), a recommendation (OME-Zarr primary, OME-TIFF export, `pyclm.io` reader for all formats), a staged path, and decisions to confirm. | Storage decisions change; the format default flips to ome-zarr (step 2d). |
| [stage3-router-design.md](stage3-router-design.md) | Stage 3 design: the Router (subscription table built from method requirements, fan-out and derived shutdown fan-in), `PipelineProcess` registration, per-experiment pattern history in the pattern process, and `TrackingProcess` as the first plug-in; implemented 2026-09-07 with every decision as recommended. §9 records the 2026-09-08 additions: named `[segmentation.<name>]` tables (kind `seg:<name>`) and the measurement toolbox (`core/measure.py`). | The router, registration or tracking surfaces change. |
| [stage4-control-plane-design.md](stage4-control-plane-design.md) | Stage 4 record: scope narrowed to values inside a fixed schedule, changed by the pattern method for its own experiment (`context.set_*`), applied at the timepoint boundary, recorded in `events.parquet` and as frames-table columns; acknowledgements, lateness and `status.json`. Implemented 2026-09-08. | The settings API, the events table or the status file change. |
| [stage5-schema-setup-design.md](stage5-schema-setup-design.md) | Stage 5 design and record: the configuration schema field by field (`schema.py`), `pyclm check`, the single `pyclm` command, `pyclm preview`, `pyclm new`; behaviour changes (unknown keys are errors, `steps`/`interval_seconds` required, per-channel overrides fixed, `t_delay` placement). Implemented 2026-09-08. | The schema, the check or the command line change. |
| [stage5b-interactivity-design.md](stage5b-interactivity-design.md) | Stage 5b options and record: napari vs ndv vs pymmcore-gui for the viewer (verified versions, compatibility measured, suite run on the upgraded stack), the command-file protocol for a running experiment. Implemented 2026-09-09; the control window was withdrawn the same day (see the end of the document). | The viewer stack or the command protocol change. |
| [testing-guide-stage5b.md](testing-guide-stage5b.md) | What to test dry and at the microscope after Stages 3–5b, step by step, including the pymmcore device-interface check the upgrade requires. | The stack or the commands change. |
| [spikes/](spikes/) | Runnable spike scripts and their artefacts. Not part of the package, not tested. | A spike is superseded (delete it) or a new one is run. |

Conventions for this folder:

- Cite code as `path:line` so a reader can jump to it. Line numbers drift; a
  function name next to the line number keeps the reference useful.
- Record *verified* facts. If something was checked empirically (a test run, a
  script), say so and say how.
- Keep opinions in the assessment/roadmap files, not in the architecture notes.
