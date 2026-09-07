# Data format and export

PyCLM writes one output per experiment (one per imaging position) while the
run is in progress, so nothing is lost if a run is aborted. Two storage
formats are available, selected in `pyclm_config.toml`:

```toml
[output]
format = "hdf5"              # "hdf5" (default) or "ome-zarr"
pattern_policy = "on_change" # ome-zarr only: "on_change", "all", or "none"
export_imagej = true         # write ImageJ hyperstacks when the run ends
```

| | `ome-zarr` (format 2) | `hdf5` (format 1) |
|---|---|---|
| Output per experiment | `<experiment>.zarr/` folder | `<experiment>.hdf5` file |
| Opens without PyCLM in | Fiji, napari, QuPath, Python (`zarr`, `dask`) | Python (`h5py`) only; layout is PyCLM-specific |
| Live viewing during a run | Yes (`--gui`, or napari on the folder) | Yes (`--gui`, via SWMR) |
| Compression | zstd per frame | none |
| Skipped timepoints | Cost nothing on disk | Pre-allocated empty datasets |
| Standard | [OME-NGFF 0.4](https://ngff.openmicroscopy.org/0.4/) on zarr v2 | PyCLM's original layout |

`hdf5` remains the default until the OME-Zarr layout has been used for a
full real experiment. Both formats are read by the same tools
(`pyclm.io`, `convert_hdf5s`, the live GUI), and files written by older
PyCLM versions stay readable.

## What an experiment directory contains after a run

```
experiment_dir/
├── bar10.pos1.zarr/          # one store per experiment (or bar10.pos1.hdf5)
├── bar025.pos1.zarr/
├── bar10.pos1_imaging.tif    # ImageJ hyperstacks, exported when the run ends
├── bar025.pos1_imaging.tif
├── frames.parquet            # (ome-zarr) one row per frame and stimulation event
├── frames.csv                # the same table as CSV, written when the run ends
├── plan.useq.yaml            # the acquisition plan derived from your TOMLs and positions
├── all_layers.txt            # layer list used by the live GUI
└── log.log
```

The plan (`plan.useq.yaml`) is a [useq-schema](https://pymmcore-plus.github.io/useq-schema/)
`MDASequence` describing every position, channel, exposure, and the timing,
with PyCLM's cadence and stimulation settings under `metadata.pyclm`. It is
also embedded in every output, so a store or file is self-describing even if
it is moved on its own.

## The OME-Zarr layout

```
bar10.pos1.zarr/
├── .zattrs                       pyclm: format, plan, experiment and schedule metadata,
│                                 affine_transform, slm_shape, groups, current_t
├── imaging/                      one OME-NGFF image per cadence group
│   ├── .zattrs                   multiscales (axes t, c, y, x; time scale in seconds), omero
│   ├── 0/                        (T, C, Y, X) uint16, one compressed chunk per frame
│   └── labels/segmentation/0/    (T, C, Y, X) uint16 label images, if segmentation is saved
└── patterns/dmd/0/               (N, H_slm, W_slm) uint8 DMD patterns, one per distinct pattern
```

### Cadence groups

Channels that share an `every_t` are stored together as one image with a
**compact** time axis: one slot per acquisition of that group, not one per
timepoint of the run. In the usual configuration, imaging channels every 5
timepoints and stimulation every timepoint with the stimulation frame not
saved, there is a single group called `imaging` whose time axis has one
fifth as many entries as the run has timepoints. Nobody sees four blank
frames between real ones.

Each group records `every_t` and `t_delay` in its `pyclm` attributes and as
the NGFF time-scale transform (`every_t × interval_seconds`), so viewers that
honour the transform (napari does) place groups of different cadence on the
same real time axis. To map a slot back to the plan timepoint:
`t = t_delay + slot × every_t`.

Group names: `imaging` holds the first imaging channel and everything on its
cadence; a group holding only the stimulation camera frame is `stim`; any
other cadence is `imaging_every<N>`. The stimulation camera frame is only
stored when `[stimulation] save = true` in the experiment TOML, and then it
is the last channel of the group of its own cadence.

### Patterns

`patterns/dmd/0` holds the DMD-space patterns that were uploaded to the
light modulator. With `pattern_policy = "on_change"` (default) each distinct
pattern is stored once; the frames table records which pattern was in force
at every stimulation event. In a closed-loop experiment this is one pattern
per imaging frame; in an open-loop experiment with a moving pattern it is
one per timepoint, because there the pattern is the stimulus. `"all"` stores
one entry per stimulation event regardless; `"none"` stores no patterns.
Patterns compress extremely well: a full 1080-timepoint run costs a few
megabytes.

The pattern in camera coordinates is not stored; it is computed on export
and in the GUI from the affine transform saved in the store's attributes.

### The frames table

`frames.parquet` (rewritten as the run proceeds) and `frames.csv` (written
at the end) have one row per acquired frame (`kind = "frame"`) and one row
per stimulation event (`kind = "stim_event"`), for all experiments in the
directory:

| Column | Meaning |
|---|---|
| `experiment` | position label, e.g. `bar10.pos1` |
| `kind` | `frame` or `stim_event` |
| `t` | plan timepoint |
| `group`, `local_index` | where the frame lives in the store (frames only) |
| `channel` | channel name (the stimulation channel's preset for stimulation events) |
| `scheduled_at`, `completed_at` | wall-clock timestamps (ISO 8601) |
| `exposure_ms`, `binning`, `pixel_size_um` | acquisition settings |
| `x`, `y`, `z`, `pfs_offset` | stage position |
| `pattern_id`, `pattern_index` | the pattern in force (index into `patterns/dmd/0`) |

### Opening OME-Zarr data

**Fiji / ImageJ.** Use *File ▸ Import ▸ HDF5/N5/Zarr/OME-NGFF* (the n5-ij
reader bundled with recent Fiji) and choose the `imaging` image inside the
`.zarr` folder, or install [MoBIE](https://mobie.github.io/). The exported
`.tif` hyperstacks (below) are the simplest route for most ImageJ work.

**napari.** Install the `napari-ome-zarr` plugin and drag the `imaging`
folder into the viewer, or use PyCLM's own viewer, which needs no plugin:

```bash
uv run gui path/to/experiment_dir
```

**Python, without PyCLM.**

```python
import zarr

store = zarr.open_group("bar10.pos1.zarr", mode="r")
img = store["imaging/0"]           # (T, C, Y, X)
frame = img[3, 0]                  # slot 3 of channel 0 as a numpy array
labels = store["imaging/labels/segmentation/0"][3, 0]
patterns = store["patterns/dmd/0"] # (N, H, W)
info = store.attrs["pyclm"]        # plan YAML, groups, affine transform, ...
```

**Python, with PyCLM.** `pyclm.io` reads both formats behind one API and
resolves the bookkeeping for you:

```python
import pyclm.io

exp = pyclm.io.open("experiment_dir/bar10.pos1.zarr")   # or .hdf5
g = exp.groups["imaging"]
g.channels                 # ('545', '638')
g.acquired()               # slots that hold data, e.g. [0, 1, 2, ...]
g.frame(3, "545")          # numpy array, or None if not acquired
g.labels(3, "545")         # segmentation labels, or None
g.global_t(3)              # the plan timepoint of slot 3
exp.pattern_at(g.global_t(3))   # DMD pattern in force at that timepoint
exp.frames                 # pyarrow.Table of the frames rows for this experiment
exp.frames.to_pandas()     # if pandas is installed
exp.affine_transform       # camera → DMD affine (2×3)
exp.plan_yaml              # the acquisition plan
```

### Copying and archiving

A `.zarr` store is a folder of many small chunk files. Copy or move the
whole folder as a unit, together with `frames.parquet` / `frames.csv` and
`plan.useq.yaml` from the same directory. If a single file is required for
archiving, zip the folder; `zarr` can read a zipped store directly.

## The HDF5 layout (format 1)

The original layout, one SWMR file per experiment:

```
bar10.pos1.hdf5
├── attrs: schedule_metadata, experiment_metadata, plan, every_t, t_delay, t_stop, t_count
├── current_t_index                 highest completed timepoint (for live readers)
├── 00000/
│   ├── channel_545/data            (Y, X) uint16, attrs: timing, position, index, ...
│   ├── channel_545/seg             segmentation labels (pre-allocated even when unused)
│   ├── stim_aq/data                camera frame during stimulation (if saved)
│   └── stim_aq/dmd                 (H_slm, W_slm) uint8 pattern, attrs: pattern_id
├── 00001/ ...
```

Every dataset for every timepoint is pre-allocated with shape `(0, 0)` when
the run starts and resized when its frame is written; a dataset that stays
`(0, 0)` was never acquired. `pyclm.io.open()` presents each channel as its
own group (`imaging_545`, `stim_aq`) with the same API as above, so analysis
code does not need to know which format it is reading.

## Exporting to ImageJ

Every export writes one hyperstack per cadence group, named
`<experiment>_<group>.tif`, with the axes `TCYX`, the frame interval and
pixel size set for ImageJ, and channels in this order:

1. the raw channels of the group, grey LUT;
2. the segmentation labels of each channel, yellow LUT (only if any were saved);
3. the DMD pattern in force at each frame, warped into camera coordinates,
   cyan LUT (only if the affine transform is known).

**Automatically.** With `export_imagej = true` (the default) the stacks are
written into the experiment directory when the run ends.

**From the command line**, for a finished directory in either format:

```bash
uv run convert_hdf5s path/to/experiment_dir
```

Restrict the export to particular channels or groups by naming them:

```bash
uv run convert_hdf5s path/to/experiment_dir 545 stim
```

Older HDF5 files that predate the embedded affine transform take it from
`pyclm_config.toml`; pass `--config path/to/pyclm_config.toml` if it is not
in the experiment directory or the working directory.

**From Python**, with control over what goes in:

```python
import pyclm.io

with pyclm.io.open("experiment_dir/bar10.pos1.zarr") as exp:
    paths = pyclm.io.export_imagej(
        exp,
        out_dir="exports",         # default: next to the data
        groups=["imaging"],        # default: every group
        overlay_pattern=True,      # add the cyan pattern channel
        include_labels=True,       # add the yellow label channels
    )
```

Frames that were never acquired are not part of the export: the hyperstack's
time axis is the group's compact axis, and `finterval` carries the real
seconds between frames.
