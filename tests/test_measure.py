"""
The measurement toolbox (pyclm.core.measure): Regions, PerTrack and the
nuclear/cytosolic ratio, plus Tracks behaving as Regions.
"""

import numpy as np
import pytest

from pyclm.core.measure import PerTrack, Regions, nuclear_cytosolic_ratio
from pyclm.core.tracking import TrackRow, Tracks


def discs(cells, shape=(64, 64), radius=4):
    img = np.zeros(shape, dtype=np.uint32)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for label, y, x in cells:
        img[(yy - y) ** 2 + (xx - x) ** 2 <= radius**2] = label
    return img


# ----------------------------------------------------------------- Regions
def test_regions_ids_areas_centroids_and_masks():
    labels = discs([(3, 10, 10), (7, 40, 30)])
    regions = Regions(labels)

    assert regions.ids.tolist() == [3, 7]
    assert len(regions) == 2
    assert regions.areas().tolist() == [
        int((labels == 3).sum()),
        int((labels == 7).sum()),
    ]
    assert np.allclose(regions.centroids(), [[10, 10], [40, 30]])
    assert regions.mask(7).sum() == (labels == 7).sum()
    assert repr(regions) == "Regions(2 objects, labels (64, 64))"


def test_regions_measure_statistics_are_aligned_with_ids():
    labels = discs([(1, 10, 10), (2, 40, 40)])
    image = np.full(labels.shape, 5.0)
    image[labels == 1] = 100.0
    image[labels == 2] = np.where(np.arange((labels == 2).sum()) % 2, 10.0, 30.0)
    regions = Regions(labels)

    assert regions.measure(image).tolist() == [100.0, pytest.approx(20.0, abs=1.0)]
    assert regions.measure(image, "max").tolist() == [100.0, 30.0]
    assert regions.measure(image, "min").tolist() == [100.0, 10.0]
    assert regions.measure(image, "sum")[0] == 100.0 * (labels == 1).sum()
    assert regions.measure(image, "std")[0] == 0.0
    with pytest.raises(ValueError, match="unknown stat"):
        regions.measure(image, "mode")
    with pytest.raises(ValueError, match="does not match"):
        regions.measure(image[:10])
    assert Regions(np.zeros((8, 8), np.uint16)).measure(np.ones((8, 8))).shape == (0,)


def test_regions_paint_from_scalar_dict_and_array():
    labels = discs([(1, 10, 10), (2, 40, 40), (5, 20, 50)])
    regions = Regions(labels)

    full = regions.paint(1.0)
    assert full.dtype == np.float32
    assert np.all(full[labels > 0] == 1.0)
    assert np.all(full[labels == 0] == 0.0)

    by_dict = regions.paint({1: 0.25, 5: 0.75, 99: 1.0})
    assert np.all(by_dict[labels == 1] == 0.25)
    assert np.all(by_dict[labels == 2] == 0.0)
    assert np.all(by_dict[labels == 5] == 0.75)

    by_array = regions.paint(np.array([0.1, 0.2, 0.3]), background=-1.0)
    assert np.allclose(by_array[labels == 2], 0.2)
    assert np.all(by_array[labels == 0] == -1.0)

    explicit = regions.paint([0.9], ids=[5])
    assert np.allclose(explicit[labels == 5], 0.9)
    assert np.all(explicit[labels == 1] == 0.0)

    with pytest.raises(ValueError, match="one value per id"):
        regions.paint([0.1, 0.2])


def test_regions_select_and_owner_of():
    cells = discs([(1, 16, 16), (2, 48, 48)], radius=10)
    nuclei = discs([(11, 16, 16), (12, 48, 48), (13, 5, 60)], radius=3)

    only_two = Regions(cells).select([2])
    assert only_two.ids.tolist() == [2]
    assert (only_two.labels == 1).sum() == 0

    owner = Regions(cells).owner_of(Regions(nuclei))
    assert owner.tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="same shape"):
        Regions(cells).owner_of(Regions(nuclei[:10]))


def test_tracks_are_regions_in_row_order():
    labels = discs([(9, 10, 10), (4, 40, 40)])
    rows = [TrackRow(4, 2, 40.0, 40.0, 5), TrackRow(9, 1, 10.0, 10.0, 5)]
    tracks = Tracks(labels, rows)
    image = np.where(labels == 9, 90.0, np.where(labels == 4, 40.0, 0.0))

    assert isinstance(tracks, Regions)
    assert tracks.ids.tolist() == [4, 9]  # row order, not sorted
    assert tracks.measure(image).tolist() == [40.0, 90.0]
    painted = tracks.paint(tracks.measure(image) / 100)
    assert np.allclose(painted[labels == 9], 0.9)
    assert tracks.select([9]).ids.tolist() == [9]


# ----------------------------------------------------------------- PerTrack
def test_pertrack_update_get_defaults_and_forget():
    memory = PerTrack()
    assert len(memory) == 0
    assert np.isnan(memory.get([1, 2])).all()
    assert np.isnan(memory.get([1], default="median")).all()

    memory.update([1, 2, 3], [10.0, 20.0, 60.0])
    assert memory.get([2, 3, 4]).tolist()[:2] == [20.0, 60.0]
    assert np.isnan(memory.get([4])[0])
    assert memory.get([4], default="median")[0] == 20.0
    assert memory.get([4], default="mean")[0] == 30.0
    assert memory.get([4], default=7)[0] == 7.0
    assert memory.known([1, 4]).tolist() == [True, False]
    assert 2 in memory
    assert memory[2] == 20.0
    assert memory.ids.tolist() == [1, 2, 3]
    assert memory.values().tolist() == [10.0, 20.0, 60.0]

    memory.update([2], [25.0])  # latest value wins
    assert memory[2] == 25.0
    memory.forget([1, 99])
    assert memory.ids.tolist() == [2, 3]
    assert memory.as_dict() == {2: 25.0, 3: 60.0}
    with pytest.raises(ValueError, match="unknown default"):
        memory.get([5], default="mode")
    memory.clear()
    assert len(memory) == 0


# ------------------------------------------------- nuclear_cytosolic_ratio
def test_nuclear_cytosolic_ratio_matches_nuclei_to_cells():
    cells = discs([(1, 16, 16), (2, 48, 48)], radius=10)
    nuclei = discs([(11, 16, 16), (12, 48, 48), (13, 60, 5)], radius=3)
    image = np.full(cells.shape, 1.0)
    image[cells == 1] = 10.0  # cytosol of cell 1
    image[cells == 2] = 20.0  # cytosol of cell 2
    image[nuclei == 11] = 30.0
    image[nuclei == 12] = 10.0
    image[nuclei == 13] = 50.0

    out = nuclear_cytosolic_ratio(nuclei, Regions(cells), image)
    assert out.nucleus_ids.tolist() == [11, 12, 13]
    assert out.cell_ids.tolist() == [1, 2, 0]
    assert out.nuclear.tolist() == [30.0, 10.0, 50.0]
    assert out.cytosolic.tolist()[:2] == [10.0, 20.0]
    assert np.isnan(out.cytosolic[2])
    assert out.ratio.tolist()[:2] == [3.0, 0.5]
    assert np.isnan(out.ratio[2])

    # the readout can be painted back on the nuclei or on the cells
    on_nuclei = Regions(nuclei).paint(np.nan_to_num(out.ratio))
    assert np.allclose(on_nuclei[nuclei == 11], 3.0)
    on_cells = Regions(cells).paint(out.ratio[:2], ids=out.cell_ids[:2])
    assert np.allclose(on_cells[cells == 2], 0.5)
