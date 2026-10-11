# PyCLM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20385964.svg)](https://doi.org/10.5281/zenodo.20385964)

A Modular Closed-Loop Microscopy Software in Python

Please see our documentation at [readthedocs](https://pyclm.readthedocs.io/en/latest/first_time_setup.html)

![](documentation/imgs/Figure%201.png "PyCLM Overview")

## Overview

PyCLM is a python-based closed-loop microscopy software designed for running many dynamic and/or feedback-controlled optogenetic experiments simultaneously. Through PyCLM, it is possible to design spatial light patterns that change over time, follow individual cell positions, or change in response to cell features.

This approach to closed-loop microscopy is oriented around three major goals:

1. **Efficiency.**
Ideally, the number of possible simultaneous experiments should be limited only by the physical constraints of the microscopy system.
- Microscope control, data export, image segmentation, and pattern generation are split into separate processes that run in parallel, minimizing unneeded microscope downtime.

2. **(Re)usability.** Re-running experiments, combining multiple experiments, or changing experimental parameters should be as quick as possible, without requiring any programming experience.
- Descriptive and intuitively organized .toml files are used for experiment configuration, and can be reused or quickly edited for later experiments.
- Several common use-case configurations are provided for dynamic light input and feedback control experiments.

3. **Extensibility.** It should be possible to implement a wide range of new segmentation and stimulation methods on top of the existing software.
- Segmentation and stimulation methods inherit from a base class, and are required to implement three straightforward functions.

## Installation

```bash
pip install "closed-loop-microscopy[gui]"
```

The package is `closed-loop-microscopy`; the import name and the command are
`pyclm`. Leave out `[gui]` for a headless install (checking, previews, dry
runs, reading data). For development, clone the repository and run
`uv sync`.

Try it without a microscope:

```bash
pyclm new demo --demo
pyclm run demo --dry --gui
```

The demo runs two experiments on the virtual microscope with synthetic cells:
a sweeping bar of light, and a closed loop that segments the cells with a
method the directory brings along (`demo_methods.py`) and lights the outward
half of each.

See [First-Time Setup Guide](https://pyclm.readthedocs.io/en/latest/first_time_setup.html) to get started.

## Setting up with an AI coding agent

New to PyCLM? The repository includes three
[agent skills](.claude/skills/): step-by-step guides that let a coding
agent such as [Claude Code](https://claude.com/claude-code) walk you
through the parts that usually take the longest. The agent runs the
`pyclm` checks itself, and asks you only about your microscope and your
experiment.

| Skill | What it helps with |
|---|---|
| [`pyclm-microscope-bringup`](.claude/skills/pyclm-microscope-bringup/SKILL.md) | Getting PyCLM running on your microscope: install, the Micro-Manager device interface, the `.cfg` presets, `pyclm_config.toml` (DMD calibration, camera ROI, focus lock), and the first real run. |
| [`pyclm-design-experiment`](.claude/skills/pyclm-design-experiment/SKILL.md) | Turning an experiment idea ("a bar of light moving at 1 µm/min over half my wells") into a checked experiment folder and a dry run on the virtual microscope. |
| [`pyclm-write-pattern-method`](.claude/skills/pyclm-write-pattern-method/SKILL.md) | Turning a light rule described in words ("light cells whose reporter is below X") into a tested pattern method. |

To use them, start Claude Code in a clone of this repository; it loads
the skills from `.claude/skills/` and the project notes from
`CLAUDE.md`. Then describe what you want, for example "help me set up
PyCLM on our Nikon Ti2 with a Mightex Polygon". To use the skills from
another folder, such as the one holding your experiments, copy the three
folders in `.claude/skills/` into that folder's `.claude/skills/`, or
into `~/.claude/skills/` to have them everywhere. The skills are plain
Markdown in the open `SKILL.md` format, so other agents that read skills,
or a person, can follow them too.

The agent never changes your Micro-Manager configuration; it tells you
what to change. Its work on the virtual microscope (`--dry`) is safe to
try before going to the scope.

### Hardware requirement: Micro-Manager

PyCLM controls hardware through [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus), which wraps the [Micro-Manager](https://micro-manager.org) C++ core.  The Python package installs without issue, but running a real experiment requires:

1. **Install Micro-Manager** — download the nightly build for your platform from [micro-manager.org](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release).
2. **Point PyCLM at your MM installation** — set the `config_path` field in `pyclm_config.toml` to your Micro-Manager `.cfg` hardware configuration file.

Importing the package and using the dry-run (`--dry`) mode work without a Micro-Manager installation.


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
