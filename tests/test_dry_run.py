# ruff: noqa: B023
"""
Dry-run integration tests.

Each test exercises the full run_pyclm pipeline against a SimulatedMicroscopeCore,
loading real experiment TOMLs and TIF stacks from tests/dry_run_resources/.

Tests cover:
  - multipoints.xml + PFSPositionMover  (explicit image source via from_tiff_stack)
  - PositionList.pos + PFSPositionMover  (explicit image source via from_tiff_stack)
  - PositionList.pos + BasicPositionMover (explicit image source via from_tiff_stack)
  - dry_run.yml auto-discovery
  - TIF-filename auto-discovery (fallback case)
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from pyclm import BasicPositionMover, PFSPositionMover, run_pyclm
from pyclm.core.virtual_microscope.simulated_source import TimeSeriesImageSource

# ---------------------------------------------------------------------------
# Paths to test resources
# ---------------------------------------------------------------------------

RESOURCES = Path(__file__).parent / "dry_run_resources"
TOMLS = RESOURCES / "tomls"
TIFS = RESOURCES / "tifs"

XML_POSITION_COUNT = 2
POS_POSITION_COUNT = 2

# ---------------------------------------------------------------------------
# Expected dataset structure (derived from test resources)
#
# test_schedule.toml  : steps=4, so t = 0, 1, 2, 3
# bar*.toml imaging   : every_t=2, save=true  → 2 channel datasets (t=0, t=2)
# bar*.toml stim      : every_t=1, save=false → 4 DMD datasets (all steps)
#
# Timepoint key format: f"{t:05d}" → "0000", "0001", …  (space-padded)
#
# Camera source: 800x800 px → stored shape (800, 800) uint16
# SLM shape from pyclm_config.toml : slm_shape_h=1140, slm_shape_w=912
# ---------------------------------------------------------------------------

_STEPS = 4
_IMAGING_EVERY_T = 2
_STIM_EVERY_T = 1
# Simulated camera returns the TIF dimensions; test TIFs are 800x800.
_CAMERA_SHAPE = (800, 800)
# Shape of datasets that are pre-allocated but never written (shape stays (0,0)).
_EMPTY_SHAPE = (0, 0)


@dataclass(frozen=True)
class _Spec:
    shape: tuple
    dtype: np.dtype


def _t_key(t: int) -> str:
    """Replicates the manager's f'{t:05d}' zero-padded format."""
    return f"{t:05d}"


# Build the canonical dataset map every output file must contain.
#
# Per-timepoint structure (from manager.py pre-allocation):
#   stim_aq/data  — written when stim.save=true  → shape _CAMERA_SHAPE
#   stim_aq/seg   — pre-allocated but never filled (no seg method) → (0, 0)
#   channel_*/data — written at imaging channel every_t            → shape _CAMERA_SHAPE
#   channel_*/seg  — pre-allocated but never filled                → (0, 0)
# No stim_aq/dmd because SimulatedMicroscopeCore is created with slm_device=None.
#
# Plus one top-level scalar: current_t_index.
EXPECTED_DATASETS: dict[str, _Spec] = {
    "current_t_index": _Spec(shape=(), dtype=np.dtype("int32")),
}

for _t in range(_STEPS):
    _k = _t_key(_t)
    if _t % _STIM_EVERY_T == 0:
        EXPECTED_DATASETS[f"{_k}/stim_aq/data"] = _Spec(
            shape=_CAMERA_SHAPE, dtype=np.dtype("uint16")
        )
        EXPECTED_DATASETS[f"{_k}/stim_aq/seg"] = _Spec(
            shape=_EMPTY_SHAPE, dtype=np.dtype("uint16")
        )
    if _t % _IMAGING_EVERY_T == 0:
        EXPECTED_DATASETS[f"{_k}/channel_545/data"] = _Spec(
            shape=_CAMERA_SHAPE, dtype=np.dtype("uint16")
        )
        EXPECTED_DATASETS[f"{_k}/channel_545/seg"] = _Spec(
            shape=_EMPTY_SHAPE, dtype=np.dtype("uint16")
        )

# ---------------------------------------------------------------------------
# Common TOML/schedule files copied into every experiment directory
# ---------------------------------------------------------------------------

_COMMON_FILES = ("bar10.toml", "bar025.toml", "pyclm_config.toml")


def _copy_common(tmp_path: Path) -> None:
    for name in _COMMON_FILES:
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")


# ---------------------------------------------------------------------------
# Fixtures — explicit image source (pos/xml tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def image_source() -> TimeSeriesImageSource:
    """Single-frame source built from the fast-bar TIF stack, used for all positions."""
    return TimeSeriesImageSource.from_tiff_stack(TIFS / "mdck_fast_bar.tif", loop=True)


@pytest.fixture
def xml_experiment_dir(tmp_path) -> Path:
    """
    Experiment directory with multipoints.xml as the position list.
    PositionList.pos is intentionally excluded so the XML fallback is exercised.
    The caller supplies an explicit image source, so no TIFs are needed here.
    """
    _copy_common(tmp_path)
    shutil.copy(TOMLS / "multipoints.xml", tmp_path / "multipoints.xml")
    return tmp_path


@pytest.fixture
def pos_experiment_dir(tmp_path) -> Path:
    """
    Experiment directory with PositionList.pos as the position list.
    multipoints.xml is intentionally excluded so .pos parsing is exercised.
    The caller supplies an explicit image source, so no TIFs are needed here.
    """
    _copy_common(tmp_path)
    shutil.copy(TOMLS / "PositionList.pos", tmp_path / "PositionList.pos")
    return tmp_path


# ---------------------------------------------------------------------------
# Fixtures — auto-discovery (yml and tif-name tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def yml_experiment_dir(tmp_path) -> Path:
    """
    Experiment directory with a dry_run.yml that maps two positions to two
    different TIF files.  No position list is present so the yml takes priority.
    """
    _copy_common(tmp_path)
    shutil.copy(TIFS / "mdck_fast_bar.tif", tmp_path / "mdck_fast_bar.tif")
    shutil.copy(TIFS / "mdck_slow_bar.tif", tmp_path / "mdck_slow_bar.tif")

    yml_content = (
        "positions:\n"
        "  - name: bar10.00\n"
        "    source: mdck_fast_bar.tif\n"
        "  - name: bar025.00\n"
        "    source: mdck_slow_bar.tif\n"
    )
    (tmp_path / "dry_run.yml").write_text(yml_content)

    return tmp_path


@pytest.fixture
def tif_name_experiment_dir(tmp_path) -> Path:
    """
    Experiment directory with no position list or dry_run.yml.  TIF filenames
    encode the position labels: bar10.00.tif → label bar10.00 → bar10.toml,
    bar025.00.tif → label bar025.00 → bar025.toml.
    """
    _copy_common(tmp_path)
    shutil.copy(TIFS / "mdck_fast_bar.tif", tmp_path / "bar10.00.tif")
    shutil.copy(TIFS / "mdck_slow_bar.tif", tmp_path / "bar025.00.tif")
    return tmp_path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def assert_hdf5_content(experiment_dir: Path, expected_file_count: int) -> None:
    """
    Assert that *expected_file_count* HDF5 files exist in *experiment_dir* and
    that each file contains exactly the datasets described by EXPECTED_DATASETS,
    with the correct shapes and dtypes.
    """
    hdf5_files = sorted(experiment_dir.glob("*.hdf5"))
    assert len(hdf5_files) == expected_file_count, (
        f"Expected {expected_file_count} HDF5 file(s), found {len(hdf5_files)}: "
        + ", ".join(f.name for f in hdf5_files)
    )

    for fp in hdf5_files:
        with h5py.File(fp, "r") as f:
            actual: dict[str, h5py.Dataset] = {}

            def collect_dsets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    actual[name] = obj

            f.visititems(collect_dsets)

            assert len(actual) == len(EXPECTED_DATASETS), (
                f"{fp.name}: expected {len(EXPECTED_DATASETS)} datasets, "
                f"got {len(actual)}.\n"
                f"  Expected: {sorted(EXPECTED_DATASETS)}\n"
                f"  Actual  : {sorted(actual)}"
            )

            for key, spec in EXPECTED_DATASETS.items():
                assert key in actual, (
                    f"{fp.name}: missing dataset '{key}'.\n"
                    f"  Present datasets: {sorted(actual)}"
                )
                dset = actual[key]
                assert dset.shape == spec.shape, (
                    f"{fp.name}['{key}']: expected shape {spec.shape}, got {dset.shape}"
                )
                assert dset.dtype == spec.dtype, (
                    f"{fp.name}['{key}']: expected dtype {spec.dtype}, got {dset.dtype}"
                )


# ---------------------------------------------------------------------------
# Tests — explicit image source (position list formats)
# ---------------------------------------------------------------------------


def test_dry_run_xml_pfs(xml_experiment_dir, image_source):
    """
    multipoints.xml position list + PFSPositionMover + explicit image source.
    """
    run_pyclm(
        xml_experiment_dir,
        position_mover=PFSPositionMover(),
        dry_image_source=image_source,
        dry=True,
    )
    assert_hdf5_content(xml_experiment_dir, XML_POSITION_COUNT)


def test_dry_run_pos_pfs(pos_experiment_dir, image_source):
    """
    PositionList.pos + PFSPositionMover + explicit image source.
    The .pos file carries PFSOffset in extras; PFSPositionMover reads and applies it.
    """
    run_pyclm(
        pos_experiment_dir,
        position_mover=PFSPositionMover(),
        dry_image_source=image_source,
        dry=True,
    )
    assert_hdf5_content(pos_experiment_dir, POS_POSITION_COUNT)


def test_dry_run_pos_basic(pos_experiment_dir, image_source):
    """
    PositionList.pos + BasicPositionMover + explicit image source.
    Exercises the same .pos parsing path as test_dry_run_pos_pfs but with a
    plain XYZ mover that ignores extras such as PFSOffset.
    """
    run_pyclm(
        pos_experiment_dir,
        position_mover=BasicPositionMover(),
        dry_image_source=image_source,
        dry=True,
    )
    assert_hdf5_content(pos_experiment_dir, POS_POSITION_COUNT)


# ---------------------------------------------------------------------------
# Tests — auto-discovery
# ---------------------------------------------------------------------------


def test_dry_run_yml(yml_experiment_dir):
    """
    dry_run.yml auto-discovery: no explicit image source or position list.
    The yml maps bar10.00 → mdck_fast_bar.tif and bar025.00 → mdck_slow_bar.tif.
    Expected output: bar10.00.hdf5 and bar025.00.hdf5.
    """
    run_pyclm(yml_experiment_dir, dry=True)
    assert_hdf5_content(yml_experiment_dir, 2)


def test_dry_run_tif_names(tif_name_experiment_dir):
    """
    TIF-filename fallback: no position list or dry_run.yml present.
    bar10.00.tif → position bar10.00 → bar10.toml,
    bar025.00.tif → position bar025.00 → bar025.toml.
    Expected output: bar10.00.hdf5 and bar025.00.hdf5.
    """
    run_pyclm(tif_name_experiment_dir, dry=True)
    assert_hdf5_content(tif_name_experiment_dir, 2)
