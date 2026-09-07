# Usage

For first time usage, see the [First-Time Setup Guide](first_time_setup.md).

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
- Experiment progress can be monitored through the experiment log, or in real time using the `--gui` flag, which opens a live Napari viewer that updates as images are acquired.

## 3. Analyzing a completed multi-experiment
Each experiment (position) is written to its own output as the run proceeds, either an OME-Zarr store or an HDF5 file depending on `[output] format` in `pyclm_config.toml`. Alongside the images the output holds the acquisition plan, per-frame timing and position, the segmentation masks, and the DMD pattern in force at every stimulation.

- When the run finishes, ImageJ hyperstacks are exported next to the data (raw channels, segmentation labels, and the pattern warped into camera space). Re-run the export at any time with `uv run convert_hdf5s <experiment_dir>`.
- Open any output in Python with `pyclm.io.open(path)`; OME-Zarr outputs also open directly in Fiji, napari, and `zarr`.
- See [Data format and export](data_format.md) for the layouts, the frames table, and the export options.
- For convenience, a script for simple tracking (or segmentation and tracking) is also available.
