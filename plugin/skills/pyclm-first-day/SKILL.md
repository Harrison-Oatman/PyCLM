---
name: pyclm-first-day
description: Take a lab from nothing to a first real PyCLM run on their microscope in one day - the order of setup, bring-up, design, run and close, what can be done before the day, and what to cut when time runs short. Use when someone is setting PyCLM up on a new microscope and wants an experiment running the same day, or asks "where do I start".
---

# First day on a new microscope

The goal of the day is one real run on the lab's own sample, with light
going where it should and data they can open. Everything else can wait
for day two. This skill only orders the work; each step is another
skill.

Keep a short running log in the session's folder (`first_day_log.md`):
the time each step started and ended, and each point where a person had
to do something by hand (move the sample, click in Micro-Manager, decide
a setting). It is the best record of what to fix for the next lab.

## Before the day (any computer, no microscope)

Done remotely or the day before, these save the first hour:

1. On the microscope PC: Claude Code installed and signed in, the plugin
   installed (`plugin/README.md`, "Install"), Python 3.11 or later on the
   `PATH` (`python --version` in the shell Claude Code uses).
2. PyCLM installed (`pyclm-microscope-bringup`, step 1) and
   `pyclm new demo --demo` then `pyclm run demo --dry` working.
3. The rig inventory (`pyclm-microscope-bringup`, step 0) written down,
   above all the light-patterning device's Micro-Manager adapter and its
   device interface, which decides the pymmcore version.
4. Micro-Manager opens the `.cfg` the lab uses, with the imaging channels
   and the light path to the patterning device as presets.

## On the day

| Step | Skill | Rough time | Done when |
| --- | --- | --- | --- |
| 1. Lab context | `pyclm-lab-setup` | 15 min | the next session's start note names the lab and the microscope |
| 2. Device interface and `.cfg` | `pyclm-microscope-bringup` 1-2 | 30 min | `pyclm check --mm-config` is clean, pixel size right |
| 3. Calibration and first light | `pyclm-microscope-bringup` 3-4 | 1-2 h | a `full_on` frame is lit edge to edge; a `bar` moves the right way |
| 4. The experiment | `pyclm-design-experiment` | 30-60 min | `pyclm check` passes and a dry run on a real frame looks right |
| 5. The run | `pyclm-run` | as designed | `pyclm watch` says done, data open |
| 6. Close | `pyclm-close-session` | 15 min | run records copied, proposals written |

At steps 4 and 5, make the Design, Positions and Running pages
(`pyclm-session-pages`) for the person to look at beside the conversation.
They are views built from files; every approval is still typed in the
conversation.

Steps 1 and 4 can overlap with the hardware steps: the person who knows
the biology can answer the design questions while the calibration slide
is on the stage.

## Rules of thumb for a same-day run

- **Calibrate on a fluorescent slide, then switch to cells.** Do first
  light (`full_on`, then `bar`) before the real sample goes on the stage.
- **Start simple.** For the first run, prefer a built-in method from the
  method zoo (`full_on`, `bar`, `global_pulse`, `dose_grid`) over a custom
  feedback rule; a custom method is day two (`pyclm-write-pattern-method`).
- **One unlit control.** Even a short first run is more useful with an
  unlit position next to the lit ones.
- **Short before long.** A 10-minute run with three timepoints per
  position on the real sample, then the real run. It catches focus, light
  and timing problems that the dry run cannot.
- **Offer a plate map on day one.** If the lab reuses a vessel format,
  ask whether the person will help set up a calibration for it
  (`pyclm-run`, "Positions, focus and plate maps"): later sessions on
  this microscope can start from it instead of searching. Write the z
  and lock offset that worked into the microscope note's "Finding focus"
  section.
- **Focus.** With no hardware focus lock, use `position_mover = "basic"`
  and few positions; check focus on the frames at the first timepoints.
  Thick samples (organoids, embryos, tissue) drift in z; say so before the
  run and watch `focus` flags.
- **Leave on time.** If calibration runs long, cut positions and duration,
  not the unlit control or the short test run.

## What to cut when time runs short

In this order: segmentation and feedback (run open-loop patterns),
extra conditions, movies (export later), the close-session proposals
(write them the next day from the log). Never cut first light: a run
with the light in the wrong place is worse than no run.

## After the day

Read `first_day_log.md` back with the person. Each hand step marked as
avoidable becomes a proposal (`pyclm-close-session`, step 2): to the
microscope note, the lab file, or an issue for PyCLM.
