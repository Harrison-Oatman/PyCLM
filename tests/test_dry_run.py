# ruff: noqa: B023
"""
Dry-run integration tests.

Each test exercises the full run_pyclm pipeline against a SimulatedMicroscopeCore,
loading real experiment TOMLs and TIF stacks from tests/dry_run_resources/.

The three tests cover the two supported position-list formats and both built-in
PositionMover implementations.
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

# The XML positions file has two positions: bar10.pos1 and bar025.pos1.
XML_POSITION_COUNT = 2

# The .pos file has two positions: bar10.pos1 and bar025.pos1.
POS_POSITION_COUNT = 2

# ---------------------------------------------------------------------------
# Expected dataset structure (derived from test resources)
#
# test_schedule.toml  : steps=4, so t = 0, 1, 2, 3
# bar*.toml imaging   : every_t=2, save=true  → 2 channel dataset (t=0, t=2)
# bar*.toml stim      : every_t=1, save=false → 4 DMD datasets (all steps)
#                       (save=false skips the raw image, but save_stim=True
#                        keeps the DMD pattern)
#
# Timepoint key format: f"{t: 05d}" → " 0000", " 0001", …  (space-padded)
#
# Camera source: 800x800 px, (virtual core does not bin) → stored shape (800, 800) uint16
# SLM shape from pyclm_config.toml   : slm_shape_h=1140, slm_shape_w=912
# ---------------------------------------------------------------------------

_STEPS = 4
_IMAGING_EVERY_T = 2
_STIM_EVERY_T = 1
_CAMERA_SHAPE = (800, 800)
_SLM_SHAPE = (1140, 912)


@dataclass(frozen=True)
class _Spec:
    shape: tuple
    dtype: np.dtype


def _timepoint_key(t: int) -> str:
    """Replicates the manager's f'{t: 05d}' format."""
    return f"{t: 05d}"


# Build the canonical dataset map that every output file should contain.
EXPECTED_DATASETS: dict[str, _Spec] = {}

for _t in range(_STEPS):
    if _t % _STIM_EVERY_T == 0:
        EXPECTED_DATASETS[f"{_timepoint_key(_t)}/stim_aq/dmd"] = _Spec(
            shape=_SLM_SHAPE, dtype=np.dtype("uint8")
        )
    if _t % _IMAGING_EVERY_T == 0:
        EXPECTED_DATASETS[f"{_timepoint_key(_t)}/channel_545/data"] = _Spec(
            shape=_CAMERA_SHAPE, dtype=np.dtype("uint16")
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_source() -> TimeSeriesImageSource:
    """Single-frame source built from the fast-bar TIF stack."""
    return TimeSeriesImageSource.from_tiff_stack(TIFS / "mdck_fast_bar.tif", loop=True)


@pytest.fixture
def xml_experiment_dir(tmp_path) -> Path:
    """
    Experiment directory with multipoints.xml as the position list.
    PositionList.pos is intentionally excluded so the XML fallback is exercised.
    """
    for name in ("bar10.toml", "bar025.toml", "multipoints.xml", "pyclm_config.toml"):
        shutil.copy(TOMLS / name, tmp_path / name)
    # Use the short test schedule (overrides the production 1440-step schedule)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    return tmp_path


@pytest.fixture
def pos_experiment_dir(tmp_path) -> Path:
    """
    Experiment directory with PositionList.pos as the position list.
    multipoints.xml is intentionally excluded so .pos parsing is exercised.
    """
    for name in ("bar10.toml", "bar025.toml", "PositionList.pos", "pyclm_config.toml"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
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
            # Collect every dataset path in this file.
            actual: dict[str, h5py.Dataset] = {}

            def collect_dsets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    actual[name] = obj

            f.visititems(collect_dsets)

            # --- count ---
            assert len(actual) == len(EXPECTED_DATASETS), (
                f"{fp.name}: expected {len(EXPECTED_DATASETS)} datasets, "
                f"got {len(actual)}.\n"
                f"  Expected: {sorted(EXPECTED_DATASETS)}\n"
                f"  Actual  : {sorted(actual)}"
            )

            # --- presence, shape, dtype ---
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
# Tests
# ---------------------------------------------------------------------------


def test_dry_run_xml_pfs(xml_experiment_dir, image_source):
    """
    multipoints.xml position list + PFSPositionMover.

    Both positions in the XML map to a TOML file, so two HDF5 files are
    produced, each with the expected dataset structure.
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
    PositionList.pos position list + PFSPositionMover.

    The .pos file carries a PFSOffset in extras for each position;
    PFSPositionMover reads it and applies the offset.
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
    PositionList.pos position list + BasicPositionMover.

    Exercises the same .pos parsing path as test_dry_run_pos_pfs but with a
    mover that performs a plain XYZ move and ignores extras such as PFSOffset.
    The output structure should be identical to the PFS variant.
    """
    run_pyclm(
        pos_experiment_dir,
        position_mover=BasicPositionMover(),
        dry_image_source=image_source,
        dry=True,
    )
    assert_hdf5_content(pos_experiment_dir, POS_POSITION_COUNT)
