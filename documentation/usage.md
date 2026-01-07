# Usage

For first time usage, see [_Installation_](installation.md).

Running an experiment with PyCLM is split into three parts:

## 1. Designing a multi-experiment
Multiple experiments can be run simultaneously using PyCLM, without any programming experience.

1. Choose an empty directory to contain the multi-experiment configuration files. This is also where PyCLM will generate the output (data and log files).
2. Copy in .toml files corresponding to the experiment(s) that will be run.
3. Write (or reuse) a schedule.toml file.
4. Choose imaging positions (using the micromanager interface), and assign an experiment to each position.

## 2. Running a multi-experiment

Once a directory for the multi-experiment has been set up, close out of all existing microscope control software, activate your venv or conda environment, and run main.py with command-line arguments.

- Data is saved continuously during the multi-experiment, and the code execution can be aborted at any time without data loss.
- Experiment progress can be monitored through the experiment log, as well as through continuous .tif export.

## 3. Analyzing a completed multi-experiment
Data is exported into separate .hdf5 files for each experiment, which alongside acquired images contain important metadata about timing and imaging configurations. These files also track the light pattern applied at each timepoint and the segmentation generated, if applicable.
- These .hdf5 files can be converted to .tif stacks, with or without pattern/segmentation overlay by running the `convert_to_tifs.py` script.
- For convenience, a script for simple tracking (or segmentation and tracking) is also available.
