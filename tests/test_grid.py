"""Grids: reading MicroManager's tile-creator positions, geometry, stitching and cutting."""

import json

import numpy as np
import pytest

from pyclm.core.experiments import MicroscopePosition
from pyclm.core.grid import GridGeometry, cut, geometry_for, group_tiles, stitch
from pyclm.directories import positions_from_pos


# ------------------------------------------------------------ a tile-creator list
def mm_entry(label, x, y, z, row, col, pfs=None):
    devices = [
        {
            "Device": {"type": "STRING", "scalar": "XYStage"},
            "Position_um": {"type": "DOUBLE", "array": [x, y]},
        },
        {
            "Device": {"type": "STRING", "scalar": "ZDrive"},
            "Position_um": {"type": "DOUBLE", "array": [z]},
        },
    ]
    if pfs is not None:
        devices.append(
            {
                "Device": {"type": "STRING", "scalar": "PFSOffset"},
                "Position_um": {"type": "DOUBLE", "array": [pfs]},
            }
        )
    return {
        "DefaultXYStage": {"type": "STRING", "scalar": "XYStage"},
        "DefaultZStage": {"type": "STRING", "scalar": "ZDrive"},
        "GridCol": {"type": "INTEGER", "scalar": col},
        "GridRow": {"type": "INTEGER", "scalar": row},
        "Label": {"type": "STRING", "scalar": label},
        "DevicePositions": {"type": "PROPERTY_MAP", "array": devices},
        "Properties": {"type": "PROPERTY_MAP", "scalar": {}},
    }


def write_list(path, entries):
    path.write_text(
        json.dumps(
            {
                "encoding": "UTF-8",
                "format": "Micro-Manager Property Map",
                "major_version": 2,
                "minor_version": 0,
                "map": {"StagePositions": {"type": "PROPERTY_MAP", "array": entries}},
            }
        )
    )
    return path


def tile_creator_grid(
    prefix="tissue-1",
    rows=2,
    cols=3,
    pitch=(500.0, 480.0),
    origin=(1000.0, -2000.0),
    z=50.0,
):
    """Tiles as MicroManager's TileCreator lays them out: snake order, labels prefix-col_row."""
    entries = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in cs:
            entries.append(
                mm_entry(
                    f"{prefix}-{c:03d}_{r:03d}",
                    origin[0] + c * pitch[0],
                    origin[1] + r * pitch[1],
                    z + r,  # interpolated focus
                    r,
                    c,
                    pfs=100.0 + c,
                )
            )
    return entries


def test_tiles_fold_into_one_grid_position(tmp_path):
    entries = [
        mm_entry("bar10-1", 0.0, 0.0, 10.0, 0, 0),
        *tile_creator_grid(),
        mm_entry("bar025-1", 9.0, 9.0, 10.0, 0, 0),
    ]
    positions = positions_from_pos(write_list(tmp_path / "PositionList.pos", entries))

    assert [p.label for p in positions] == ["bar10.1", "tissue.1", "bar025.1"]
    plain, grid, _ = positions
    assert not plain.is_grid
    assert grid.is_grid
    assert grid.grid == {"rows": 2, "columns": 3, "pitch_um": [500.0, 480.0]}
    assert len(grid.tiles) == 6
    assert [t.grid_rc for t in grid.tiles] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 1),
        (1, 0),
    ]  # snake
    assert grid.tiles[3].label == "tissue.1.002_001"
    # the grid sits at the centre of its tiles, its extras averaged
    assert (grid.x, grid.y) == pytest.approx((1000.0 + 500.0, -2000.0 + 240.0))
    assert grid.z == pytest.approx(50.5)
    assert grid.extras["PFSOffset"] == pytest.approx(101.0)
    assert grid.tiles[1].extras["PFSOffset"] == 101.0


def test_grid_with_a_deleted_tile_or_uneven_spacing_is_refused(tmp_path):
    entries = tile_creator_grid()[:-1]
    with pytest.raises(ValueError, match="complete 2 x 3 rectangle"):
        positions_from_pos(write_list(tmp_path / "a.pos", entries))

    entries = tile_creator_grid()
    entries[1]["DevicePositions"]["array"][0]["Position_um"]["array"][0] += 60.0
    with pytest.raises(ValueError, match="not uniform"):
        positions_from_pos(write_list(tmp_path / "b.pos", entries))


def test_grid_fields_that_contradict_the_label_are_refused():
    tiles = [MicroscopePosition(0, 0, 0), MicroscopePosition(1, 0, 0)]
    with pytest.raises(ValueError, match="disagree"):
        group_tiles(tiles, ["g-1-000_000", "g-1-001_000"], [(0, 0), (1, 1)])


# ------------------------------------------------------------ geometry, stitch, cut
def test_geometry_from_the_position_and_the_camera(tmp_path):
    grid = positions_from_pos(
        write_list(tmp_path / "PositionList.pos", tile_creator_grid())
    )[0]
    assert geometry_for(MicroscopePosition(0, 0, 0), (0, 0, 100, 80), 1.0) is None

    # 1 um / px, ROI 500 x 480: tiles abut exactly
    g = geometry_for(grid, (0, 0, 500, 480), 1.0)
    assert (g.rows, g.columns) == (2, 3)
    assert g.pitch(1) == (480, 500)
    assert g.overlap(1) == (0, 0)
    assert g.shape(1) == (960, 1500)
    assert g.shape(2) == (480, 750)
    assert g.tiles == ((0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0))

    # a larger ROI than the pitch overlaps, a smaller one leaves gaps
    assert geometry_for(grid, (0, 0, 520, 500), 1.0).overlap(1) == (20, 20)
    assert geometry_for(grid, (0, 0, 400, 400), 1.0).overlap(1) == (-80, -100)


def test_stitch_places_tiles_by_row_and_column_and_cut_inverts_it():
    g = GridGeometry(2, 2, 10.0, 8.0, 8, 10, ((0, 0), (0, 1), (1, 1), (1, 0)))
    frames = [np.full((8, 10), v, np.uint16) for v in (1, 2, 3, 4)]
    canvas = stitch(frames, g, 1)
    assert canvas.shape == (16, 20)
    assert canvas[0, 0] == 1
    assert canvas[0, 19] == 2
    assert canvas[15, 19] == 3
    assert canvas[15, 0] == 4

    tiles = cut(canvas, g, 1)
    assert [int(t.max()) for t in tiles] == [1, 2, 3, 4]
    assert all(t.shape == (8, 10) for t in tiles)

    # binning 2: pitch and tile sizes halve
    small = [np.full((4, 5), v, np.uint8) for v in (1, 2, 3, 4)]
    assert stitch(small, g, 2).shape == (8, 10)

    # overlap: the later tile overwrites the strip; gaps stay zero
    g2 = GridGeometry(1, 2, 8.0, 8.0, 8, 10, ((0, 0), (0, 1)))
    canvas = stitch(
        [np.full((8, 10), 1, np.uint8), np.full((8, 10), 2, np.uint8)], g2, 1
    )
    assert canvas.shape == (8, 18)
    assert canvas[0, 8] == 2
    g3 = GridGeometry(1, 2, 12.0, 8.0, 8, 10, ((0, 0), (0, 1)))
    canvas = stitch(
        [np.full((8, 10), 1, np.uint8), np.full((8, 10), 2, np.uint8)], g3, 1
    )
    assert canvas.shape == (8, 22)
    assert canvas[0, 10] == 0
    assert canvas[0, 12] == 2
