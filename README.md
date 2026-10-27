# PyCLM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20385964.svg)](https://doi.org/10.5281/zenodo.20385964)
[![Documentation](https://readthedocs.org/projects/pyclm/badge/?version=latest)](https://pyclm.readthedocs.io/en/latest/)

A modular closed-loop microscopy software in Python.

PyCLM runs many optogenetic experiments at once on one microscope. Each
imaging position gets its own light pattern, projected through a DMD, that
can be fixed, change over time, follow individual cells, or respond to what
the cells are doing. Experiments are written as short TOML files, so running
a new one needs no programming; a new kind of light rule is a small Python
class.

![](https://raw.githubusercontent.com/Harrison-Oatman/PyCLM/main/documentation/imgs/pyclm-overview.png "PyCLM Overview")

## Start here

| I want to... | Go to |
|---|---|
| See what light patterns PyCLM can make | **[Method zoo](https://pyclm.readthedocs.io/en/latest/method_zoo.html)**: every built-in pattern method on a sample image, with a `[pattern]` table to copy |
| Try it without a microscope | The demo below |
| Set PyCLM up on my microscope | [First-time setup](https://pyclm.readthedocs.io/en/latest/first_time_setup.html) |
| Write an experiment | [Experiment TOMLs](https://pyclm.readthedocs.io/en/latest/experiment_tomls.html) and [the `pyclm` command](https://pyclm.readthedocs.io/en/latest/command_line.html) |
| Write my own light rule | [Custom pattern methods](https://pyclm.readthedocs.io/en/latest/custom_pattern_methods.html) |
| Open the data from a run | [Data format and export](https://pyclm.readthedocs.io/en/latest/data_format.html) |

The full documentation is at [pyclm.readthedocs.io](https://pyclm.readthedocs.io/en/latest/).

## Install and try it

```bash
pip install "closed-loop-microscopy[gui]"
pyclm new demo --demo
pyclm run demo --dry --gui
```

The package is `closed-loop-microscopy`; the import name and the command are
`pyclm`. Leave out `[gui]` for a headless install (checking, previews, dry
runs, reading data). For development, clone the repository and run `uv sync`.

The demo runs two experiments on the virtual microscope (`--dry`) with
synthetic cells: a sweeping bar of light, and a closed loop that segments the
cells and lights the outward half of each. Pick a method from the
[zoo](https://pyclm.readthedocs.io/en/latest/method_zoo.html), paste its
`[pattern]` table over the one in `demo/cells.toml` (keep `channel = "545"`,
the demo's channel), and run it again.

Running on a real microscope needs [Micro-Manager](https://micro-manager.org),
which PyCLM drives through
[pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus); point
`config_path` in `pyclm_config.toml` at your Micro-Manager `.cfg` file. The
[installation page](https://pyclm.readthedocs.io/en/latest/installation.html)
covers the extras and matching the Micro-Manager device interface.

## How it works

Microscope control, segmentation, tracking, pattern generation and saving run
as separate processes in parallel, so the microscope rarely waits and the
number of simultaneous experiments is limited mainly by the hardware. Data is
written as the run goes (OME-Zarr by default), and ImageJ hyperstacks are
exported at the end.

## Setting up with an AI coding agent

The repository includes three [agent skills](https://github.com/Harrison-Oatman/PyCLM/tree/main/.claude/skills):
step-by-step guides a coding agent such as
[Claude Code](https://claude.com/claude-code) can follow to walk you through
the slow parts. The agent runs the `pyclm` checks itself and asks you only
about your microscope and your experiment.

| Skill | What it helps with |
|---|---|
| [`pyclm-microscope-bringup`](https://github.com/Harrison-Oatman/PyCLM/blob/main/.claude/skills/pyclm-microscope-bringup/SKILL.md) | Getting PyCLM running on your microscope: install, the Micro-Manager device interface, the `.cfg` presets, `pyclm_config.toml` (DMD calibration, camera ROI, focus lock), and the first real run. |
| [`pyclm-design-experiment`](https://github.com/Harrison-Oatman/PyCLM/blob/main/.claude/skills/pyclm-design-experiment/SKILL.md) | Turning an experiment idea ("a bar of light moving at 1 µm/min over half my wells") into a checked experiment folder and a dry run on the virtual microscope. |
| [`pyclm-write-pattern-method`](https://github.com/Harrison-Oatman/PyCLM/blob/main/.claude/skills/pyclm-write-pattern-method/SKILL.md) | Turning a light rule described in words ("light cells whose reporter is below X") into a tested pattern method. |

Start Claude Code in a clone of this repository and describe what you want,
for example "help me set up PyCLM on our Nikon Ti2 with a Mightex Polygon".
To use the skills from another folder, copy the folders in `.claude/skills/`
into that folder's `.claude/skills/`, or into `~/.claude/skills/`. They are
plain Markdown (`SKILL.md`), so other agents, or a person, can follow them
too. The agent never changes your Micro-Manager configuration; it tells you
what to change.

## Citation

If you use PyCLM in your research, please cite:

> Oatman HR, Lad BC, Nguyen L, Gubbens JH, Toettcher JE. An optogenetic smart
> microscopy platform reveals signaling dynamics-dependent control over
> collective cell migration. *Cell Systems* 17(9) (2026).
> [doi:10.1016/j.cels.2026.101674](https://doi.org/10.1016/j.cels.2026.101674)

The paper describes version 1.0.1. To record the version you used, also cite
that release: every release has its own DOI on Zenodo, all listed under
[doi:10.5281/zenodo.20385964](https://doi.org/10.5281/zenodo.20385964)
(which always resolves to the latest). GitHub's "Cite this repository" button
gives both, from `CITATION.cff`.
