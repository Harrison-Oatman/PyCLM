"""
The shared Qt widgets (status line, minimap) with an offscreen Qt platform:
no napari, no display.
"""

import json
import os

import pytest
from helpers import make_experiment, make_plan, make_schedule

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qtpy = pytest.importorskip("qtpy")
from qtpy import QtWidgets

from pyclm.gui.widgets import (
    MapPosition,
    Minimap,
    RunOverview,
    StatusLine,
    positions_from_directory,
    status_text,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_status_text_covers_the_states():
    assert status_text(None) == "no status yet"
    base = {"t": 3, "timepoints": 10, "experiments": {}}
    assert status_text(base) == "t 3/10"
    assert status_text({**base, "done": True}) == "t 3/10 | done"
    assert status_text({**base, "paused": True, "current_experiment": "a.00"}) == (
        "t 3/10 | paused | at a.00"
    )
    full = {
        **base,
        "experiments": {
            "a.00": {"lateness_s": 2.5, "errors": 1},
            "b.00": {"errors": 1},
        },
        "settings_applied": 2,
        "pending_commands": 1,
    }
    assert status_text(full) == (
        "t 3/10 | a.00 late 2.5s | 2 acquisition errors | 2 settings changed | 1 command pending"
    )


def test_minimap_draws_positions_current_and_problems(app):
    m = Minimap()
    m.set_positions(
        [
            MapPosition("a.00", "a", 0.0, 0.0),
            MapPosition("a.01", "a", 1000.0, 0.0),
            MapPosition("b.00", "b", 0.0, 800.0, fov_um=(200.0, 300.0)),
        ]
    )
    # three markers, three labels, a scale bar and its label
    assert len(m.scene().items()) == 8
    assert m.colour_of("a") != m.colour_of("b")
    assert m.colour_of("a") == m.colour_of("a")

    m.set_status({"current_experiment": "a.01", "experiments": {"b.00": {"errors": 2}}})
    assert m.current == "a.01"
    assert m.problems == {"b.00"}
    assert len(m.scene().items()) == 8

    m.set_positions([])
    assert m.scene().items()[0].toPlainText() == "no positions"


def test_run_overview_reads_positions_and_status(app, tmp_path):
    exp = make_experiment("bar10.00")
    plan = make_plan(make_schedule([exp]))
    plan.to_yaml(tmp_path / "plan.useq.yaml")
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "t": 1,
                "timepoints": 3,
                "current_experiment": "bar10.00",
                "experiments": {},
            }
        )
    )

    positions = positions_from_directory(tmp_path)
    assert [(p.label, p.experiment) for p in positions] == [("bar10.00", "bar10")]

    overview = RunOverview(tmp_path)
    status = overview.refresh()
    assert status["t"] == 1
    assert overview.status.text() == "t 1/3 | at bar10.00"
    assert overview.minimap.current == "bar10.00"
    assert [p.label for p in overview.minimap.positions] == ["bar10.00"]


def test_positions_fall_back_to_the_position_list(tmp_path):
    import shutil
    from pathlib import Path

    resources = Path(__file__).parent / "dry_run_resources" / "tomls"
    shutil.copy(resources / "PositionList.pos", tmp_path / "PositionList.pos")
    labels = sorted(p.label for p in positions_from_directory(tmp_path))
    assert labels == ["bar025.pos1", "bar10.pos1"]
    assert positions_from_directory(tmp_path / "empty") == []


def test_status_line_widget(app):
    line = StatusLine()
    assert line.update_from({"t": 0, "timepoints": 2, "experiments": {}}) == "t 0/2"
    assert line.text() == "t 0/2"
