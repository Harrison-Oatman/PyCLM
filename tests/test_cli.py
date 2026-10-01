"""
The pyclm command: subcommands dispatch, the old `pyclm <dir>` form means
run, `new` writes a template directory that `check` understands, and
`export` on a directory without outputs is harmless.
"""

import pytest

from pyclm import cli
from pyclm.templates import TEMPLATES, create


def test_old_form_and_subcommands_dispatch(monkeypatch, tmp_path):
    calls = []
    for name in (
        "cmd_run",
        "cmd_check",
        "cmd_preview",
        "cmd_export",
        "cmd_gui",
        "cmd_new",
    ):
        monkeypatch.setattr(
            cli, name, lambda args, name=name: calls.append((name, vars(args))) or 0
        )
    # the parser binds functions at build time, so rebuild through main()
    assert cli.main([str(tmp_path), "--dry", "--gui"]) == 0
    assert cli.main(["run", str(tmp_path), "--force"]) == 0
    assert cli.main(["check", str(tmp_path), "--mm-config", "x.cfg"]) == 0
    assert (
        cli.main(["preview", str(tmp_path), "bar10", "--image", "a.tif", "--t", "3"])
        == 0
    )
    assert cli.main(["export", str(tmp_path), "545", "stim"]) == 0
    assert cli.main(["gui", str(tmp_path)]) == 0
    assert (
        cli.main(["new", str(tmp_path), "--template", "open-loop", "--name", "bar"])
        == 0
    )

    names = [c[0] for c in calls]
    assert names == [
        "cmd_run",
        "cmd_run",
        "cmd_check",
        "cmd_preview",
        "cmd_export",
        "cmd_gui",
        "cmd_new",
    ]
    first = calls[0][1]
    assert (first["dry"], first["gui"], first["no_check"], first["force"]) == (
        True,
        True,
        False,
        False,
    )
    assert calls[1][1]["force"] is True
    assert calls[2][1]["mm_config"] == "x.cfg"
    assert (calls[3][1]["experiment"], calls[3][1]["image"], calls[3][1]["t"]) == (
        "bar10",
        "a.tif",
        3,
    )
    assert calls[4][1]["channels"] == ["545", "stim"]
    assert (calls[6][1]["template"], calls[6][1]["name"]) == ("open-loop", "bar")


def test_no_check_and_force_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path), "--no-check", "--force"])


def test_new_writes_a_template_that_check_understands(tmp_path, capsys):
    target = tmp_path / "exp"
    assert (
        cli.main(["new", str(target), "--template", "closed-loop", "--name", "cells"])
        == 0
    )
    names = sorted(p.name for p in target.iterdir())
    assert names == ["README.txt", "cells.toml", "pyclm_config.toml", "schedule.toml"]
    assert "pyclm check" in (target / "README.txt").read_text()

    # the files parse; check finds only what the user still has to supply
    from pyclm.check import check_directory
    from pyclm.schema import ExperimentConfig, PyclmConfig, ScheduleConfig

    cfg = ExperimentConfig.from_file(target / "cells.toml")
    assert cfg.pattern.method == "move_out"
    assert cfg.segmentation.method == "cellpose"
    ScheduleConfig.from_file(target / "schedule.toml")
    PyclmConfig.from_file(target / "pyclm_config.toml")
    report = check_directory(target)
    errors = [str(f) for f in report.errors]
    assert len(errors) == 1
    assert "positions: no PositionList.pos or multipoints.xml" in errors[0]

    # nothing is overwritten
    (target / "cells.toml").write_text("edited")
    assert create(target, "closed-loop", "cells") == []
    assert (target / "cells.toml").read_text() == "edited"
    assert set(TEMPLATES) == {"open-loop", "closed-loop"}
    with pytest.raises(ValueError, match="plain word"):
        create(tmp_path / "x", "open-loop", "a.b")


def test_open_loop_template_needs_no_segmentation(tmp_path):
    from pyclm.schema import ExperimentConfig

    create(tmp_path, "open-loop", "bar")
    cfg = ExperimentConfig.from_file(tmp_path / "bar.toml")
    assert cfg.segmentation is None
    assert cfg.pattern.method == "bar"
    assert cfg.pattern.kwargs == {"duty_cycle": 0.2, "bar_speed": 1.0, "period": 100}


def test_export_on_an_empty_directory(tmp_path, capsys):
    assert cli.main(["export", str(tmp_path)]) == 0
    assert "exported 0 stack(s)" in capsys.readouterr().out


def test_demo_runs_on_the_virtual_microscope(tmp_path, capsys):
    """pyclm new --demo: a directory that checks clean and dry-runs, cells lit on their outer half."""
    import numpy as np
    import tifffile
    from skimage.filters import threshold_otsu

    import pyclm.io as pio
    from pyclm import run_pyclm
    from pyclm.check import check_directory

    target = tmp_path / "demo"
    assert cli.main(["new", str(target), "--demo"]) == 0
    names = {p.name for p in target.iterdir()}
    assert {
        "bar.toml",
        "cells.toml",
        "demo_methods.py",
        "bar.tif",
        "cells.tif",
    } <= names

    report = check_directory(target)
    assert report.ok, report.text()
    assert "segmentation method 'threshold' from demo_methods.py" in report.text()

    # a shorter schedule for the test
    (target / "schedule.toml").write_text(
        "[timing]\nsteps = 4\ninterval_seconds = 0.5\nsetup_time_seconds = 0.0\n"
        "time_between_positions = 0.05\n"
    )
    run_pyclm(target, dry=True)
    movie = tifffile.imread(target / "cells.tif")
    with pio.open(target / "cells.00.zarr") as exp:
        cam = exp.camera_pattern_at(exp.current_t) > 0
        dmd = exp.pattern_at(exp.current_t) > 0
    assert cam.shape == movie.shape[1:]  # the pattern is at the frames' binning
    # the virtual microscope advances its TIF with every snap: compare with the
    # frame the pattern was made from, the one it overlaps best
    cells = max(
        (frame > threshold_otsu(frame) for frame in movie),
        key=lambda mask: (cam & mask).sum(),
    )
    assert (cam & cells).sum() / cam.sum() > 0.9  # the light lands on cells
    assert 0.3 < (cam & cells).sum() / cells.sum() < 0.7  # about half of each
    assert abs(dmd.mean() - cam.mean()) < 0.03  # the DMD shows the same amount
