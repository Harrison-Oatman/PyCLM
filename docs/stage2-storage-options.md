# Stage 2: data storage — options and a recommendation

Date: 2026-09-07. Status: **implemented** on branch `STage2-storage` the same
day, with all eight decisions taken as recommended (1 layout, 2 cadence
groups, 3 one store per experiment, 4 Parquet plus CSV, 5 zstd, 6 export on
by default, 7 DMD patterns on change, 8 zarr v2 / NGFF 0.4). Code:
`src/pyclm/core/storage/` (`FrameWriter`, `HDF5WriterV1`, `OMEZarrWriter`,
`cadence_groups`), `src/pyclm/io/` (readers for both formats, ImageJ export),
the GUI rewritten on `pyclm.io`, `[output]` in `pyclm_config.toml`. Tests:
`tests/test_storage.py` and a dry run in the zarr format. Step 2d: after the
format was validated on the microscope (2026-09-07) the default flipped to
`ome-zarr` in Stage 3; `hdf5` remains selectable and readable.

The worry driving this stage is usability: PyCLM must not produce data that
people cannot open without PyCLM. This document sets a concrete usability
bar, describes what today's HDF5 layout does and does not deliver against it,
lays out four options, and recommends one with a staged path that keeps every
existing file readable.

---

## 1. Where the data actually goes today

Facts from the repository, not opinions:

- **Nothing analyses the HDF5 files directly.** Every notebook in `figures/`
  and every script in `scripts/` and `other/` reads TIFF stacks (`tifffile`,
  `imread`) and CSV track tables. The only readers of the `.hdf5` files are
  PyCLM's own code: the live GUI, `convert_hdf5s.py`, and `PatternReview`.
- **The real deliverable is therefore the ImageJ hyperstack** that
  `convert_hdf5s` writes: `(T, C, Y, X)` uint16 with raw, segmentation and
  the DMD pattern warped back into camera space as channels, plus ImageJ LUT
  metadata. HDF5 is an intermediate format that only PyCLM understands.
- **What HDF5 currently gives PyCLM:** one file per experiment; live reading
  while writing (SWMR); per-frame provenance in attributes; the plan and
  configuration embedded; random access to any frame. It does **not** use
  compression (`chunks=True`, no `compression=`), so a 1080-timepoint,
  2-channel, 2048² run is ~17 GB per experiment on disk.
- **What HDF5 costs PyCLM:** the schema must be pre-allocated before SWMR
  starts (every-t/channel dataset, empty `(0, 0)` placeholders, no runtime
  changes, no variable-length tables); Windows file locking forces the GUI to
  reopen the file on every refresh and forced `HDF5_USE_FILE_LOCKING=FALSE`
  globally; the layout is bespoke, so no external tool opens it meaningfully;
  and the converter must exist for anyone to see the data.

The stress is justified, but the diagnosis is specific: the problem is not
"HDF5", it is "a private layout inside HDF5 plus a converter that only PyCLM
maintains".

---

## 2. The usability bar

Whatever is chosen should satisfy all of these, and the options below are
rated against them:

| # | Requirement |
|---|---|
| U1 | Opens in **Fiji/ImageJ** as a hyperstack with correct axes, without PyCLM installed |
| U2 | Opens in **napari** by drag-and-drop, including while the experiment is running |
| U3 | Readable from **plain Python in a few lines** (numpy/zarr/tifffile), no PyCLM import |
| U4 | **Self-describing**: axes, channel names, pixel size, timestamps, the plan and per-frame provenance travel with the data |
| U5 | **Durable**: a documented public specification, expected to be readable in ten years |
| U6 | **Copyable as a unit** with ordinary tools (one folder or one file per experiment) |
| U7 | **Live-appendable** during the run, with the writer never blocked by readers |
| U8 | Stores the closed-loop extras: segmentation labels, camera-space pattern, DMD-space pattern, `pattern_id` per frame |
| U9 | Compresses well and does not spend disk on frames that were never acquired |

---

## 3. Options

### Option A — HDF5 "v2": fix the layout, keep the container

Per channel one growable dataset `channel_545/data` shaped `(T, Y, X)` with
`maxshape (None, Y, X)`, chunked per frame and compressed; `seg`, `pattern`
and `dmd` alongside; one growable compound `frames` table (t, channel,
scheduled/completed times, position, pattern_id, exposure, binning) instead of
per-dataset attributes; plan YAML and config in root attrs; SWMR retained.

| Bar | Result |
|---|---|
| U1 Fiji | **No.** Fiji's HDF5 plugin (`HDF5_Vibez`) opens datasets one at a time, no hyperstack semantics; converter still required. |
| U2 napari | Only with a custom reader plugin PyCLM would have to write and maintain. |
| U3 Python | Yes (h5py), once the layout is documented. |
| U4 | Yes, in PyCLM's own attribute conventions. |
| U5 | HDF5 itself is durable; the layout is private. |
| U6 | Yes, one file. |
| U7 | Yes, SWMR; Windows locking workaround stays. |
| U8 | Yes. |
| U9 | Yes with compression; unacquired frames still occupy chunks unless written sparsely. |

Effort ~1 week. Keeps the smallest change surface, but leaves the core
problem (private layout, mandatory converter, Windows locking) in place.

### Option B — OME-Zarr (OME-NGFF) as the primary format

One NGFF image group per experiment: `bar10.pos1.zarr/` with a `(T, C, Y, X)`
array (chunk = one frame, compressed), `labels/segmentation` as an NGFF
labels image, `patterns/camera` and `patterns/dmd` as additional arrays, and
`.zattrs` holding the NGFF multiscales/omero metadata plus PyCLM's plan and
frames table (or the frames table as a Parquet sidecar).

| Bar | Result |
|---|---|
| U1 Fiji | **Yes** via the official OME-NGFF Fiji/BioFormats support (`ngff` reader, MoBIE); drag-and-drop opens the hyperstack. |
| U2 napari | **Yes**, natively (`napari-ome-zarr`), including live: napari re-reads chunks, no locking. |
| U3 Python | **Yes**: `zarr.open(path)["0"][t, c]`, or `xarray`/`dask` lazily. |
| U4 | Yes, by specification (axes, units, channel names, coordinate transforms) plus PyCLM extras in attrs. |
| U5 | Public spec maintained by OME, adopted by OMERO, QuPath, Fiji, napari, webKnossos, and the pymmcore ecosystem. |
| U6 | One **folder** per experiment (thousands of chunk files). Copy with any tool; zip at the end if a single file is wanted (a zipped store is still readable by zarr, but not live-writable). Zarr v3 sharding reduces file counts if needed. |
| U7 | **Yes**, and simpler than SWMR: appending along T is writing new chunk files; readers never block writers; nothing platform-specific. |
| U8 | Yes: labels are first-class in NGFF; extra arrays are ordinary. |
| U9 | Yes; unacquired frames cost nothing (chunk never written, fill value on read). |

Dependencies: `zarr` (not installed; `tensorstore` already is via
pymmcore-plus). pymmcore-plus ships an `OMEZarrWriter` and a
`TensorStoreHandler` keyed on the useq event index that Stage 1 introduced;
either can be used as a reference or reused for the raw frames. Effort ~2
weeks including the GUI switching to a zarr reader and the export command.

Caveats to state honestly: a folder of many small files is slower to copy
across some network shares and to back up with tools that count files;
NGFF version churn (0.4 vs 0.5 on zarr v3) means pinning a version and
recording it; and per-frame provenance is not part of the NGFF spec, so
PyCLM defines that piece (as it does today).

### Option C — OME-TIFF as the primary format

Write each experiment as an OME-TIFF (BigTIFF) hyperstack directly, with
segmentation and patterns as extra channels or companion files.

| Bar | Result |
|---|---|
| U1 Fiji | **Yes**, the native format. |
| U2 napari | Yes (tifffile reader); live viewing is fragile because the file is being appended. |
| U3 Python | Yes (`tifffile.imread`). |
| U4 | Yes (OME-XML), though PyCLM's per-frame extras need custom annotations. |
| U5 | Yes. |
| U6 | One file. |
| U7 | **Weak.** Appending planes to a TIFF during a run is possible with tifffile but readers see an incomplete/uncertain file until the OME-XML header is finalised; SWMR-style live viewing is not a supported use. |
| U8 | Awkward: DMD-space patterns have a different shape from the camera frames, so they need a second file. |
| U9 | Compression yes; unacquired frames must be written as planes. |

Best as the **export** format, not the live one.

### Option D — Keep HDF5 v1 as is, export automatically at the end

Add an automatic `export` step producing ImageJ TIFFs when a run finishes.
Cheapest (days), fixes the "people cannot open it" symptom for finished runs,
changes nothing about the fixed schema, Windows locking, uncompressed files,
or live viewing. A stop-gap, not a plan.

---

## 4. Recommendation

**Option B (OME-Zarr) as the live and archival format, with OME-TIFF export
for the ImageJ workflow, and a small `pyclm.io` package that reads every
format PyCLM has ever written.** Rationale:

1. It is the only option that meets the Fiji, napari and plain-Python bars
   **without** a PyCLM-maintained converter in the critical path. The
   converter becomes a convenience, not a prerequisite for seeing data.
2. It keeps every benefit the lab gets from HDF5 today (chunked random
   access, compression, one container per experiment, embedded metadata) and
   removes its two real liabilities: the pre-allocated schema and Windows
   SWMR locking. Runtime schedule edits (Stage 4) and tracking tables (Stage
   3) need a growable, schema-light store; zarr is that by construction.
3. It lines up with Stage 1: pymmcore-plus's writers already map the useq
   event index onto NGFF axes, so PyCLM's `index = {t, p, c}` becomes the
   array index directly, and z/grid axes arrive with the plan.
4. Legacy files stay usable: `pyclm.io` reads the v1 HDF5 layout forever,
   and `convert_hdf5s` keeps working on them.

Why not A: it spends a week making a private layout tidier while leaving
people dependent on the converter. Why not C as primary: live appending is
the one thing OME-TIFF is bad at, and live viewing is a feature the lab uses.

### Proposed layout (per experiment)

```
experiment_dir/
├── plan.useq.yaml
├── bar10.pos1.zarr/
│   ├── .zattrs                 NGFF multiscales + omero (channel names, LUTs),
│   │                           pyclm: {format: 2, plan (yaml), experiment, schedule}
│   ├── imaging/0               one NGFF image per cadence group: (T, C, Y, X) uint16,
│   │                           chunks (1, 1, Y, X), compressed, compact T with a time-scale
│   │                           transform (every_t × interval); the saved stimulation camera
│   │                           frame, if enabled, is the last channel of its group
│   ├── imaging/labels/segmentation/0   (T, C, Y, X) uint16 label images (NGFF labels)
│   ├── patterns/dmd/0          (N, H_slm, W_slm) uint8, one per distinct pattern_id
│   │                           (save-on-change; see decision 8)
│   └── frames.parquet (or .csv) one row per acquired frame and per stimulation event:
│                               t, group, channel, scheduled_at, completed_at, exposure_ms,
│                               binning, x, y, z, pfs_offset, pattern_id, pixel_size_um, errors
└── log.log
```

Channels are grouped by cadence (decision 2): in the common configuration
that is a single `imaging/` group, and no separate stimulation stack exists.
Within a group T is compact (one slot per acquisition of that group), with
`every_t` and `t_delay` in the group attrs and as an NGFF time-scale
transform, so napari overlays groups of different cadence on real time and
nobody sees blank frames. The `frames` table is the authority on what was
actually acquired and when, and which pattern was applied at every
stimulation event.

### `pyclm.io`

One module with a stable, PyCLM-independent-shaped API:

```python
exp = pyclm.io.open("bar10.pos1.zarr")      # or "bar10.pos1.hdf5" (v1)
exp.raw[t, "545"]        # numpy frame;  exp.raw.dask for lazy stacks
exp.labels[t, "545"]; exp.pattern.camera[t]; exp.pattern.dmd[t]
exp.frames               # pandas DataFrame of the frames table
exp.plan                 # AcquisitionPlan (from the embedded YAML)
pyclm.io.export_imagej(exp, "bar10.pos1.tif", channels=[...], overlay_pattern=True)
```

`convert_hdf5s` becomes a thin CLI over `export_imagej` that accepts both
formats. The GUI reads through `pyclm.io` too, which also removes the
duplicated helper functions (known-issues #16).

---

## 5. Staged path (each step shippable, dry-run tests green)

| Step | What | Size |
|---|---|---|
| 2a | Introduce a `FrameWriter` interface behind `MicroscopeOutbox` (`open(plan)`, `write(frame)`, `write_labels`, `write_pattern`, `close`); move today's HDF5 code into `HDF5WriterV1` unchanged. Add `pyclm.io` with the v1 reader and `export_imagej`, and switch the GUI and converter to it. | ~4 days |
| 2b | `OMEZarrWriter` implementing the layout above (own code on `zarr`/`tensorstore`, using pymmcore-plus's writer as reference). `[output] format = "ome-zarr" \| "hdf5"` in `pyclm_config.toml`, default **hdf5** until validated. Dry-run tests run both writers against the same expected inventory. | ~1 week |
| 2c | GUI reads zarr live (napari-ome-zarr or `pyclm.io`); Windows locking code deleted for the zarr path. Automatic ImageJ export at the end of a run (opt-in flag). | ~3 days |
| 2d | After one real experiment has been run and analysed end to end on zarr: flip the default to `ome-zarr`, keep `hdf5` available for one release, then deprecate the v1 writer (reader stays forever). | after a real run |

---

## 6. Decisions to confirm

1. **Primary format: OME-Zarr** (recommended), or HDF5 v2 if a single file
   per experiment outweighs native tool support.
2. **How channels with different cadences share a time axis.** A single
   dense `(T, C, Y, X)` array shows fill-value (black) frames to every
   generic reader for skipped timepoints: with imaging at `every_t = 5` and
   stimulation every timepoint, 80 % of imaging slots are blank in Fiji and
   napari. Revised recommendation: **one image per cadence group** (channels
   sharing `every_t` form one `(T, C, Y, X)` array with a compact time axis
   and an NGFF time-scale transform, so napari overlays them on real time;
   in the common configuration that is `imaging/` and `stim/`), with
   `every_t`/`t_delay` in the group attrs and the frames table, and the
   ImageJ export offering "hold last frame" or "drop to imaging cadence"
   when merging groups. One array per channel is the fallback if per-channel
   binning is ever allowed.
3. **One zarr group per experiment/position** (recommended; matches one HDF5
   per experiment today) versus one store per run containing all positions.
4. **Frames table as Parquet** (pandas/pyarrow, typed) versus CSV (universal,
   untyped). Recommended: Parquet with a CSV copy written at the end of the run.
5. **Compression**: zstd via blosc (recommended) or none, to be measured on a
   real fluorescence stack before choosing the level.
6. **Automatic ImageJ export at end of run**: on by default, or opt-in.
8. **Stimulation data.** Three separate things: the light-delivery event
   (record only its `pattern_id` in the frames table), the camera frame
   taken during stimulation (off by default as today; when saved it is the
   last channel of the cadence group whose timepoint it coincides with, so no
   separate stimulation stack exists), and the DMD pattern array.
   Recommended pattern policy: **save on change**, one entry per distinct
   `pattern_id` in `patterns/dmd`, keyed from the frames table. In closed
   loop that is one pattern per imaging frame (the user's proposal); in open
   loop with a moving pattern it keeps every timepoint, because there the
   pattern *is* the stimulus. Measured cost with chunk compression: a
   1080-timepoint run of 1140×912 uint8 patterns is ~1 MB (bar) to ~9 MB
   (per-cell), versus 1.1 GB uncompressed today. Config:
   `[stimulation] save_pattern = "on_change" | "all" | "none"`, default
   `on_change`. The camera-space projection is computed at export rather
   than stored (or stored per distinct pattern, to be decided).
7. **Zarr version**: v2 + NGFF 0.4 today (widest reader support, what
   pymmcore-plus writes) or v3 + NGFF 0.5 (sharding, newer). Recommended: v2
   / 0.4 now, recorded in attrs, migrate when Fiji and napari readers for 0.5
   are mainstream.
