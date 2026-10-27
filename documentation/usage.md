# Usage

A PyCLM run is one directory: the files that describe the experiments, and the
data the run writes next to them. For the first time on a new microscope, see
the [First-Time Setup Guide](first_time_setup.md).

## 1. Design the experiments

Several experiments run at once, one per imaging position, without any
programming.

1. Start a directory with `pyclm new <dir>` (`--template open-loop` or
   `closed-loop`), or copy one from an earlier run.
2. Write one `<experiment>.toml` per experiment: its channels, stimulation,
   and the `[pattern]` table that picks the light rule. The
   [Method zoo](method_zoo.md) has a table to copy for every built-in method;
   [Experiment TOMLs](experiment_tomls.md) explains every key.
3. Write (or reuse) a `schedule.toml`: how many timepoints, and how far apart.
4. Choose the imaging positions in Micro-Manager and save the position list
   into the directory as `multipoints.xml` (or `PositionList.pos`). A position named
   `<experiment>.<suffix>` runs `<experiment>.toml`.

## 2. Check, preview, run

```bash
pyclm check <dir>                                # every problem in the files
pyclm preview <dir> <experiment> --image frame.tif  # what the pattern does on an image
pyclm run <dir> --dry --gui                      # the whole run on the virtual microscope
pyclm run <dir> --gui                            # the real run
```

`pyclm run` checks again and refuses to start on errors. Close other
microscope control software before a real run. Data is saved as the run goes,
so stopping it (Ctrl+C, or a `stop_run` command) loses nothing already
acquired; `--gui` opens a live viewer. All the subcommands, and the commands
you can send to a running experiment, are in [The pyclm command](command_line.md).

## 3. Analyze the data

Each experiment (position) is written to its own output as the run proceeds,
an OME-Zarr store by default or an HDF5 file with `[output] format = "hdf5"`
in `pyclm_config.toml`. Next to the images it holds the acquisition plan,
per-frame timing and position, the segmentations and tracks, and the DMD
pattern in force at every stimulation.

- When the run finishes, ImageJ hyperstacks are exported next to the data
  (raw channels, segmentation labels, and the pattern warped into camera
  space). Export again at any time with `pyclm export <dir>`.
- Open any output in Python with `pyclm.io.open(path)`; OME-Zarr outputs also
  open directly in Fiji, napari, and `zarr`.
- See [Data format and export](data_format.md) for the layouts, the frames
  and tracks tables, and the export options.
