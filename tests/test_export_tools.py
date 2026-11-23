"""
`pyclm export --movies`, `pyclm pack` and `pyclm stitch` on a finished demo
run: one movie per position, a packed copy that verifies, catches damage and
unpacks to the same files, and an overview with every field.
"""

import json
import os
import time
import zipfile

import numpy as np
import pytest

from pyclm import cli
from pyclm.pack import pack, unpack, verify
from pyclm.stitch import ORIENTATIONS, Field, newest_fields, overview


@pytest.fixture(scope="module")
def finished_demo(tmp_path_factory):
    from pyclm.demo import create_demo
    from pyclm.run_pyclm import run_pyclm

    demo = tmp_path_factory.mktemp("runs") / "demo"
    create_demo(demo)
    run_pyclm(demo, dry=True)
    _age(demo)
    return demo


def _age(directory):
    """As if the run had ended an hour ago (pack waits for files to settle)."""
    old = time.time() - 3600
    for p in directory.rglob("*"):
        if p.is_file():
            os.utime(p, (old, old))


def test_movies(finished_demo, tmp_path):
    import cv2

    from pyclm import io as pyclm_io

    out = tmp_path / "movies"
    assert cli.main(["export", str(finished_demo), "--movies", "--out", str(out)]) == 0
    names = {p.name.rsplit(".", 1)[0] for p in pyclm_io.find_experiments(finished_demo)}
    movies = sorted(out.glob("*.mp4"))
    assert {m.stem for m in movies} == names
    cap = cv2.VideoCapture(str(movies[0]))
    ok, frame = cap.read()
    assert ok
    assert frame.ndim == 3
    assert not list(out.glob("*.part.mp4"))


def test_movies_split(finished_demo, tmp_path):
    out = tmp_path / "split"
    assert (
        cli.main(
            ["export", str(finished_demo), "--movies", "--split", "--out", str(out)]
        )
        == 0
    )
    assert all("_" in p.stem for p in out.glob("*.mp4"))


def test_stitch(finished_demo, tmp_path):
    import cv2

    fields, missing = newest_fields(finished_demo)
    assert fields, missing
    out = tmp_path / "stitched"
    assert cli.main(["stitch", str(finished_demo), "--out", str(out)]) == 0
    img = cv2.imread(str(out / "overview.png"))
    assert img is not None
    assert img.shape[2] == 3


def test_overview_places_fields_by_stage():
    a = Field("a", np.full((10, 10), 100, np.uint16), 0.0, 0.0, 1.0, 0)
    b = Field("b", np.full((10, 10), 4000, np.uint16), 100.0, 0.0, 1.0, 0)
    img = overview([a, b], max_px=1000)
    row = img[img.shape[0] // 2]
    # the bright field is to the right (stage +x), the dim one to the left
    assert row[: img.shape[1] // 2, 0].mean() < row[img.shape[1] // 2 :, 0].mean()
    assert set(ORIENTATIONS) >= {"identity", "flip_ud", "rot90"}


def test_pack_verify_unpack(finished_demo, tmp_path):
    dest = tmp_path / "copy"
    _age(finished_demo)
    assert pack(finished_demo, dest, log=print) == 0
    copy = dest / finished_demo.name
    zips = sorted(copy.glob("*.zip"))
    assert zips, "each .zarr store becomes one zip"
    manifest = json.loads((copy / "manifest.json").read_text())
    assert any(e.get("archive") for e in manifest["files"])
    assert verify(copy) == []
    assert cli.main(["pack", "--verify", str(copy)]) == 0

    out = tmp_path / "unpacked"
    assert unpack(copy, out, log=lambda *_: None) == 0
    original = {
        p.relative_to(finished_demo).as_posix(): p.stat().st_size
        for p in finished_demo.rglob("*")
        if p.is_file()
    }
    unpacked = {
        p.relative_to(out / finished_demo.name).as_posix(): p.stat().st_size
        for p in (out / finished_demo.name).rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    copied = {k: v for k, v in original.items() if k in unpacked}
    assert copied == unpacked
    skipped = set(original) - set(unpacked)
    assert all(k.endswith((".tif", ".pyc")) for k in skipped), skipped

    # damage: a truncated archive and an extra file are both caught
    with open(zips[0], "r+b") as f:
        f.truncate(zips[0].stat().st_size - 10)
    (copy / "stray.txt").write_text("x")
    problems = verify(copy)
    assert any("size differs" in p for p in problems)
    assert any("stray.txt" in p for p in problems)
    assert cli.main(["pack", "--verify", str(copy)]) == 1


def test_pack_refuses_unfinished(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"done": False, "t": 1}))
    assert pack(run, tmp_path / "dest", log=lambda *_: None) == 2
    assert pack(run, tmp_path / "dest", force=True, log=lambda *_: None) == 0
    copy = tmp_path / "dest" / "run"
    assert verify(copy) == []
    assert pack(run, tmp_path / "dest", force=True, log=lambda *_: None) == 2


def test_pack_no_zip(finished_demo, tmp_path):
    _age(finished_demo)
    assert pack(finished_demo, tmp_path, zip_folders=False, log=lambda *_: None) == 0
    copy = tmp_path / finished_demo.name
    assert not list(copy.glob("*.zip"))
    assert verify(copy, hashes=True) == []
    assert not zipfile.is_zipfile(copy / "status.json")
