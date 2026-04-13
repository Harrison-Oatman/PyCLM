# PyCLM
A Modular Closed-Loop Microscopy Software in Python

**Documentation available at [readthedocs](https://pyclm.readthedocs.io/en/latest/)**

![](documentation\imgs\Figure%201.png "PyCLM Overview")

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

PyCLM is installed with [uv](https://docs.astral.sh/uv/).

```bash
uv add closed-loop-microscopy
```

Optional extras:

```bash
uv add "closed-loop-microscopy[cellpose]"      # CellposeSAM segmentation (requires CUDA)
uv add "closed-loop-microscopy[calibration]"   # DMD calibration tools
```

### Hardware requirement: Micro-Manager

PyCLM controls hardware through [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus), which wraps the [Micro-Manager](https://micro-manager.org) C++ core.  The Python package installs without issue, but running a real experiment requires:

1. **Install Micro-Manager** — download the nightly build for your platform from [micro-manager.org](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release).
2. **Point PyCLM at your MM installation** — set the `config_path` field in `pyclm_config.toml` to your Micro-Manager `.cfg` hardware configuration file.

Importing the package and using the dry-run (`--dry`) mode work without a Micro-Manager installation.

