# PyCLM
A Modular Closed-Loop Microscopy Software in Python

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


## Usage
For first time usage, see [_Installation_](#installation) (below). For a more detailed overview, see [Usage](documentation/usage.md). Running an experiment with PyCLM is split into three parts:

### 1. Designing a multi-experiment:
Multiple experiments can be run simultaneously using PyCLM, without any programming experience. 

1. Choose an empty directory to contain the multi-experiment configuration files. This is also where PyCLM will generate the output (data and log files).
2. Copy in .toml files corresponding to the experiment(s) that will be run.
3. Write (or reuse) a schedule.toml file.
4. Choose imaging positions (using the micromanager interface), and assign an experiment to each position.

### 2. Running a multi-experiment:

Once a directory for the multi-experiment has been set up, close out of all existing microscope control software, activate your venv or conda environment, and run main.py with command-line arguments.

- Data is saved continuously during the multi-experiment, and the code execution can be aborted at any time without data loss.
- Experiment progress can be monitored through the experiment log, as well as through continuous .tif export.

### 3. Analyzing a completed multi-experiment:
Data is exported into separate .hdf5 files for each experiment, which alongside acquired images contain important metadata about timing and imaging configurations. These files also track the light pattern applied at each timepoint and the segmentation generated, if applicable. 
- These .hdf5 files can be converted to .tif stacks, with or without pattern/segmentation overlay by running the `convert_to_tifs.py` script.
- For convenience, a script for simple tracking (or segmentation and tracking) is also available.

### (Bonus: Reusing experiments)
Ease of reusability is a major advantage of PyCLM. {experiment}.toml files as well as schedule.toml files can be quickly copied over from old experiments to new ones. Running old experiments on a new day is as simple as copying over old files and picking out imaging positions. The details of past experiments are contained in the structure of the experiment setup.

## Installation
_Documentation is currently incomplete. Insterested users are recommended to check back in over the next week as the documentation is continually updated._

Installation can be technically demanding, and ease of installation will depend on your microscope setup. PyCLM is tested on a Nikon Ti2-Eclipse confocal microscope with a Mightex Polygon 1000 DMD, running on a Windows OS. 

### 1. Install and configure MicroManager
PyCLM does not communicate with your microscope hardware directly. Instead, it operates through pymmcore-plus, which is built on top of [MicroManager](https://micro-manager.org/), a well-documented microscope control platform with support for a wide range of devices. MicroManager software will be used to determine imaging configurations, and define config groups. It is worth familiarizing yourself with MicroManager before using PyCLM.

### 2. Create a Python virtual environment
Clone this GitHub repository to a directory where you can run code:

``git clone https://github.com/Harrison-Oatman/PyCLM.git``

### 3. Install and configure pymmcore-plus


### 4. Install CellposeSAM

### 5. Calibrate your DMD
For PyCLM to deliver spatially modulated stimuli, the relationship between DMD pixel coordinates and camera pixel coordinates needs to be determined. MicroManager provides a tool for calibrating the DMD using the "[Projector](https://micro-manager.org/Projector_Plugin)" plugin. This plugin will generate a set of DMD patterns to approximate the affine transform to/from the DMD.

### 6. Update pyclm_config.toml
``pyclm_config.toml`` is a configuration file used during every run of PyCLM to get important information about the DMD, including the DMD shape, and the a


