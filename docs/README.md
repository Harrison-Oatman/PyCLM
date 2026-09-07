# docs/ — engineering notes for PyCLM

This folder holds **internal engineering notes** about the PyCLM codebase: how the
core pipeline is wired, what is known to be fragile, and what the plan is for
changing it. It is written for developers and for AI agents working on the repo.

It is deliberately separate from `documentation/`, which is the Sphinx source for
the **user-facing** docs published on readthedocs. Nothing in this folder is
built or published.

| File | What it is | Update when |
|---|---|---|
| [architecture-notes.md](architecture-notes.md) | Factual map of the runtime: processes, queues, message and data types, timing model, shutdown protocol, HDF5 layout, coordinate frames, config parsing. No opinions. | Any change to `src/pyclm/core/`, `controller.py`, `directories.py`, or the HDF5 layout. |
| [known-issues.md](known-issues.md) | Concrete bugs, hazards, and smells found in the code, each with a file reference and a suggested fix. Numbers are stable (the assessment cites them). | An issue is fixed (collapse it to a one-line **Fixed** note, keep the number) or a new one is found (append). |
| [assessment-2026-09.md](assessment-2026-09.md) | The September 2026 architecture assessment: what works, what limits the five planned features (tracking, runtime edits, z-stacks, grids, interactive setup), and a staged refactor roadmap. | Roadmap decisions change. Keep as a dated record; write a new dated file for a later assessment rather than rewriting history. |
| [stage1-plan-design.md](stage1-plan-design.md) | Stage 1 evaluation: can useq's `MDASequence` be PyCLM's acquisition plan (yes, with a thin wrapper), what was verified, the proposed `AcquisitionPlan` design, and decisions to confirm. | Stage 1 decisions are made or the plan class lands. |
| [spikes/](spikes/) | Runnable spike scripts and their artefacts. Not part of the package, not tested. | A spike is superseded (delete it) or a new one is run. |

Conventions for this folder:

- Cite code as `path:line` so a reader can jump to it. Line numbers drift; a
  function name next to the line number keeps the reference useful.
- Record *verified* facts. If something was checked empirically (a test run, a
  script), say so and say how.
- Keep opinions in the assessment/roadmap files, not in the architecture notes.
