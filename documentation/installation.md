# Installation

PyCLM is published on PyPI as `closed-loop-microscopy`; the import name and
the command are `pyclm`.

```bash
pip install "closed-loop-microscopy[gui]"      # the microscope PC: runs and the live viewer
pip install closed-loop-microscopy             # headless: check, preview, dry runs, reading data
```

Extras: `gui` (the napari viewer), `cellpose` (CellposeSAM segmentation;
on Windows add `--extra-index-url https://download.pytorch.org/whl/cu128`
for GPU torch), `calibration` (pycromanager). With
[uv](https://docs.astral.sh/uv/), `uv tool install "closed-loop-microscopy[gui]"`
puts `pyclm` on the PATH.

For development, or to run the version the lab's microscope is locked to,
clone the repository; `uv sync` installs everything including the viewer and
the test tools:

```bash
git clone https://github.com/Harrison-Oatman/PyCLM.git
cd PyCLM
uv sync                      # add --extra cellpose for CellposeSAM
```

**Device interface.** pymmcore must speak the same Micro-Manager device
interface as your installed device adapters; `pyclm check` prints both. A
pip install gets the newest pymmcore. If your adapters are older (the
Mightex Polygon DLL is interface 71), pin the pair:
`pip install "pymmcore==11.2.1.71.0" "pymmcore-plus==0.14.0"`. The
repository's `uv.lock` already holds that pair.

Next: [First-time setup](first_time_setup.md).
