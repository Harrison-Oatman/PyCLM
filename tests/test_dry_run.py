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
#   stim_aq/dmd
#   channel_*/data — written at imaging channel every_t            → shape _CAMERA_SHAPE
#   channel_*/seg  — pre-allocated but never filled                → (0, 0)
#
# Plus one top-level scalar: current_t_index.
EXPECTED_DATASETS: dict[str, _Spec] = {
    "current_t_index": _Spec(shape=(), dtype=np.dtype("int32")),
}

_SLM_SHAPE = (1140, 912)

for _t in range(_STEPS):
    _k = _t_key(_t)
    if _t % _STIM_EVERY_T == 0:
        EXPECTED_DATASETS[f"{_k}/stim_aq/data"] = _Spec(
            shape=_CAMERA_SHAPE, dtype=np.dtype("uint16")
        )
        EXPECTED_DATASETS[f"{_k}/stim_aq/seg"] = _Spec(
            shape=_EMPTY_SHAPE, dtype=np.dtype("uint16")
        )
        EXPECTED_DATASETS[f"{_k}/stim_aq/dmd"] = _Spec(
            shape=_SLM_SHAPE, dtype=np.dtype("uint8")
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


def test_dry_run_ome_zarr(yml_experiment_dir):
    """
    Same run as test_dry_run_yml with [output] format = "ome-zarr": two zarr
    stores, no HDF5, the plan embedded, and ImageJ stacks exported at the end.
    """
    config = yml_experiment_dir / "pyclm_config.toml"
    config.write_text(
        config.read_text().replace('format = "hdf5"', 'format = "ome-zarr"')
    )
    run_pyclm(yml_experiment_dir, dry=True)

    import pyclm.io as pio

    stores = sorted(yml_experiment_dir.glob("*.zarr"))
    assert [s.name for s in stores] == ["bar025.00.zarr", "bar10.00.zarr"]
    assert not list(yml_experiment_dir.glob("*.hdf5"))
    assert (yml_experiment_dir / "frames.parquet").exists()

    for store in stores:
        with pio.open(store) as exp:
            assert exp.format == 2
            g = exp.groups["imaging"]
            assert g.every_t == _IMAGING_EVERY_T
            assert g.acquired() == list(range(_STEPS // _IMAGING_EVERY_T))
            assert g.frame(0, "545").shape == _CAMERA_SHAPE
            assert exp.current_t == _STEPS - 1
            assert exp.pattern_at(_STEPS - 1).shape == _SLM_SHAPE
        assert (yml_experiment_dir / f"{store.stem}_imaging.tif").exists()


def test_dry_run_tif_names(tif_name_experiment_dir):
    """
    TIF-filename fallback: no position list or dry_run.yml present.
    bar10.00.tif → position bar10.00 → bar10.toml,
    bar025.00.tif → position bar025.00 → bar025.toml.
    Expected output: bar10.00.hdf5 and bar025.00.hdf5.
    """
    run_pyclm(tif_name_experiment_dir, dry=True)
    assert_hdf5_content(tif_name_experiment_dir, 2)


def test_dry_run_tracking_ome_zarr(yml_experiment_dir):
    """
    A closed loop with tracking: [segmentation] and [tracking] in the TOMLs and
    a pattern method that asks for tracks. On OME-Zarr the store gains
    labels/tracks and tracks.parquet, the routing table is recorded, and the
    ImageJ export carries the tracked labels.
    """
    import pyclm.io as pio
    from pyclm.core.patterns import PatternMethod
    from pyclm.core.segmentation import SegmentationMethod

    class ThresholdSegmentation(SegmentationMethod):
        name = "threshold"

        def segment(self, data):
            from skimage.measure import label

            return label(data > np.percentile(data, 99)).astype(np.uint16)

    class FollowTracks(PatternMethod):
        name = "follow_tracks"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_requirement("545", tracks=True)

        def generate(self, context):
            tracks = context.tracks("545")
            assert tracks is not None
            return (tracks.labels > 0).astype(np.float32)

    config = yml_experiment_dir / "pyclm_config.toml"
    config.write_text(
        config.read_text().replace('format = "hdf5"', 'format = "ome-zarr"')
    )
    for name in ("bar10", "bar025"):
        toml = yml_experiment_dir / f"{name}.toml"
        text = toml.read_text()
        toml.write_text(
            text[: text.index("[pattern]")] + '[segmentation]\nmethod = "threshold"\n\n'
            '[tracking]\nmethod = "centroid"\nmax_distance_um = 30\n\n'
            '[pattern]\nmethod = "follow_tracks"\n'
        )

    run_pyclm(
        yml_experiment_dir,
        dry=True,
        segmentation_methods={"threshold": ThresholdSegmentation},
        pattern_methods={"follow_tracks": FollowTracks},
    )

    assert (yml_experiment_dir / "tracks.parquet").exists()
    stores = sorted(yml_experiment_dir.glob("*.zarr"))
    assert len(stores) == 2
    for store in stores:
        with pio.open(store) as exp:
            g = exp.groups["imaging"]
            assert g.has_labels
            assert g.has_tracks
            assert g.acquired() == list(range(_STEPS // _IMAGING_EVERY_T))
            assert g.tracks(0, "545").shape == _CAMERA_SHAPE
            assert g.tracks(0, "545").max() > 0
            assert exp.tracks is not None
            assert exp.tracks.num_rows > 0
            routes = exp.routing["routes"][exp.name]["545"]
            assert routes["raw"] == ["segmentation", "writer(record)"]
            assert routes["seg"] == ["tracking", "writer(record)"]
            assert routes["tracks"] == ["pattern@pattern", "writer(record)"]
        import tifffile

        stack = tifffile.imread(yml_experiment_dir / f"{store.stem}_imaging.tif")
        assert stack.shape[1] == 4  # raw, segmentation, tracks, pattern


def test_dry_run_named_segmentations_ome_zarr(yml_experiment_dir):
    """
    Two [segmentation] tables of one channel (the default and "bright"), a
    pattern that asks for both: the store carries one label image per
    table, the routing table shows both kinds, and the export has both
    label sets.
    """
    import pyclm.io as pio
    from pyclm.core.patterns import PatternMethod
    from pyclm.core.segmentation import SegmentationMethod

    class Threshold(SegmentationMethod):
        name = "threshold"

        def __init__(self, experiment_name, percentile=99, **kwargs):
            super().__init__(experiment_name, **kwargs)
            self.percentile = percentile

        def segment(self, data):
            from skimage.measure import label

            return label(data > np.percentile(data, self.percentile)).astype(np.uint16)

    class TwoSegmentations(PatternMethod):
        name = "two_segmentations"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_requirement("545", seg=["segmentation", "bright"])

        def generate(self, context):
            both = context.regions("545").labels > 0
            both &= context.regions("545", "bright").labels > 0
            return both.astype(np.float32)

    config = yml_experiment_dir / "pyclm_config.toml"
    config.write_text(
        config.read_text().replace('format = "hdf5"', 'format = "ome-zarr"')
    )
    for name in ("bar10", "bar025"):
        toml = yml_experiment_dir / f"{name}.toml"
        text = toml.read_text()
        toml.write_text(
            text[: text.index("[pattern]")]
            + '[segmentation]\nmethod = "threshold"\npercentile = 95\n\n'
            '[segmentation.bright]\nmethod = "threshold"\npercentile = 99.5\n\n'
            '[pattern]\nmethod = "two_segmentations"\n'
        )

    run_pyclm(
        yml_experiment_dir,
        dry=True,
        segmentation_methods={"threshold": Threshold},
        pattern_methods={"two_segmentations": TwoSegmentations},
    )

    import tifffile

    for store in sorted(yml_experiment_dir.glob("*.zarr")):
        with pio.open(store) as exp:
            g = exp.groups["imaging"]
            assert g.label_names == ("segmentation", "bright")
            for i in g.acquired():
                assert g.labels(i, "545").max() > 0
                assert g.labels(i, "545", "bright").max() > 0
                # the bright threshold keeps fewer pixels than the default
                assert (g.labels(i, "545", "bright") > 0).sum() < (
                    g.labels(i, "545") > 0
                ).sum()
            routes = exp.routing["routes"][exp.name]["545"]
            assert routes["seg"] == ["pattern@pattern", "writer(record)"]
            assert routes["seg:bright"] == ["pattern@pattern", "writer(record)"]
        stack = tifffile.imread(yml_experiment_dir / f"{store.stem}_imaging.tif")
        assert stack.shape[1] == 4  # raw, two label sets, pattern
