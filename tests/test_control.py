"""
The control window, offscreen: the run panel (check, commands), the
positions panel on the virtual microscope (add, save, load), the
position-list writer, and the forms (round trip through tomlkit, validation).
"""

import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qtpy = pytest.importorskip("qtpy")
from qtpy import QtWidgets

from pyclm.commands import pending
from pyclm.core.experiments import MicroscopePosition
from pyclm.directories import positions_from_pos, write_position_list
from pyclm.gui.control import ControlWindow
from pyclm.gui.forms import FileForm, write_toml_preserving
from pyclm.schema import ExperimentConfig, ScheduleConfig

RESOURCES = Path(__file__).parent / "dry_run_resources"
TOMLS = RESOURCES / "tomls"


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def exp_dir(tmp_path):
    for name in ("bar10.toml", "bar025.toml", "pyclm_config.toml", "PositionList.pos"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    for tif in (RESOURCES / "tifs").glob("*.tif"):
        shutil.copy(tif, tmp_path / tif.name)
    (tmp_path / "dry_run.yml").write_text(
        "positions:\n  - name: bar10.00\n    source: mdck_fast_bar.tif\n"
        "  - name: bar025.00\n    source: mdck_slow_bar.tif\n"
    )
    return tmp_path


# -------------------------------------------------------- position list
def test_position_list_round_trip(tmp_path):
    positions = [
        MicroscopePosition(10.0, 20.0, 30.0, label="a.00", extras={"PFSOffset": 5.5}),
        MicroscopePosition(-1.5, 2.5, 0.0, label="b.01"),
    ]
    path = write_position_list(tmp_path / "PositionList.pos", positions, "XY", "Z")
    back = positions_from_pos(str(path))
    assert [(p.label, p.x, p.y, p.z) for p in back] == [
        ("a.00", 10.0, 20.0, 30.0),
        ("b.01", -1.5, 2.5, 0.0),
    ]
    assert back[0].extras == {"PFSOffset": 5.5}
    assert back[1].extras == {}


# ------------------------------------------------------------ run panel
def test_run_panel_checks_and_writes_commands(app, exp_dir):
    window = ControlWindow(exp_dir, dry=True, attach_core=False)
    panel = window.run_panel
    assert [panel.experiment.itemText(i) for i in range(panel.experiment.count())] == [
        "bar10.pos1",
        "bar025.pos1",
    ]
    assert [panel.channel.itemText(i) for i in range(panel.channel.count())] == [
        "545",
        "stimulation",
    ]
    report = panel.check()
    assert report.ok
    assert "pyclm check" in panel.report.toPlainText()
    assert not panel.run_active
    assert not panel.pause_button.isEnabled()

    panel.exposure.setValue(80)
    panel.send_exposure()
    panel.param_key.setText("bar_speed")
    panel.param_value.setText("0.5")
    panel.send_pattern()
    panel.send({"command": "pause"})
    assert (
        panel.send({"command": "set_exposure"}) is None
    )  # refused before it is written

    found = pending(exp_dir)
    assert [c.command for _, c, _ in found] == ["set_exposure", "set_pattern", "pause"]
    assert found[0][1].ms == 80.0
    assert found[1][1].parameters == {"bar_speed": 0.5}
    assert "command refused" in panel.log.toPlainText()


# ------------------------------------------------------ positions panel
def test_positions_panel_on_the_virtual_microscope(app, exp_dir):
    window = ControlWindow(exp_dir, dry=True)
    panel = window.positions_panel
    assert panel.stage is not None
    assert panel.experiments() == ["bar025", "bar10"]

    panel.load()
    assert [r["label"] for r in panel.rows()] == ["bar10.pos1", "bar025.pos1"]

    panel.stage.move_to(123.0, 456.0, 7.0)
    panel.add_current()
    rows = panel.rows()
    assert (
        rows[-1]["label"] == "bar025.00"
    )  # the first experiment when none is selected
    assert (rows[-1]["x"], rows[-1]["y"], rows[-1]["z"]) == ("123.00", "456.00", "7.00")

    panel.table.selectRow(2)
    panel.table.cellWidget(2, 1).setCurrentText("bar10")
    assert panel.rows()[2]["label"] == "bar10.00"
    panel.add_current()
    assert panel.rows()[-1]["label"] == "bar10.01"

    path = panel.save()
    back = positions_from_pos(str(path))
    assert [p.label for p in back] == [
        "bar10.pos1",
        "bar025.pos1",
        "bar10.00",
        "bar10.01",
    ]
    assert (back[2].x, back[2].y, back[2].z) == (123.0, 456.0, 7.0)
    assert "saved 4 positions" in panel.message.text()

    panel.table.selectRow(0)
    panel.move_to_selected()
    assert panel.stage.position()[0] == pytest.approx(back[0].x)
    panel.remove_selected()
    assert len(panel.rows()) == 3

    window.release_core()
    assert panel.stage is None
    assert not panel.add_button.isEnabled()


def test_preview_here_writes_a_preview(app, exp_dir):
    window = ControlWindow(exp_dir, dry=True)
    panel = window.positions_panel
    panel.load()
    panel.table.selectRow(0)
    panel.preview_here()
    assert "preview of bar10.pos1" in panel.message.text()
    assert (exp_dir / "preview" / "bar10.pos1" / "pattern_camera.tif").exists()


# ----------------------------------------------------------------- forms
def test_experiment_form_round_trips_and_validates(app, exp_dir):
    form = FileForm(exp_dir / "bar10.toml", ExperimentConfig)
    assert form.validate() == []
    value = form.document_value()
    assert value["pattern"]["method"] == "bar"
    assert value["pattern"]["bar_speed"] == 1.0
    assert value["channels"]["presets"] == ["545"]
    assert "segmentation" not in value

    # change a value through the widgets, save, and read the file back
    imaging = form.form.widgets["imaging"].form.widgets
    imaging["exposure"].setValue(75)
    pattern = form.form.widgets["pattern"]
    pattern.kwargs.add_row("period", "120")
    assert form.save()
    cfg = ExperimentConfig.from_file(exp_dir / "bar10.toml")
    assert cfg.imaging.exposure == 75.0
    assert cfg.pattern.kwargs["period"] == 120
    text = (exp_dir / "bar10.toml").read_text()
    assert "# determines channel axis" in text  # comments of untouched tables survive

    imaging["every_t"].setValue(1)
    pattern.method.setCurrentText("barr")
    problems = form.validate()
    assert problems == []  # the schema accepts any method name; check catches it
    imaging["exposure"].setValue(imaging["exposure"].minimum())
    form.form.widgets["channels"].form.widgets["presets"].setText("")
    assert any("presets" in p for p in form.validate())


def test_schedule_form_and_new_file(app, exp_dir):
    form = FileForm(exp_dir / "schedule.toml", ScheduleConfig)
    timing = form.form.widgets["timing"].form.widgets
    assert timing["steps"].value() == 4
    timing["steps"].setValue(12)
    assert form.save()
    assert ScheduleConfig.from_file(exp_dir / "schedule.toml").timing.steps == 12

    window = ControlWindow(exp_dir, dry=True, attach_core=False)
    files = window.files_panel
    assert [files.tabs.tabText(i) for i in range(files.tabs.count())] == [
        "pyclm_config.toml",
        "schedule.toml",
        "bar025.toml",
        "bar10.toml",
    ]
    files.new_name.setText("cells")
    files.new_experiment()
    assert (exp_dir / "cells.toml").exists()
    assert files.tabs.tabText(files.tabs.currentIndex()) == "cells.toml"
    ExperimentConfig.from_file(exp_dir / "cells.toml")


def test_write_toml_preserving_keeps_untouched_comments(tmp_path):
    path = tmp_path / "x.toml"
    path.write_text("# top\n[a]\n# keep me\nk = 1\n\n[b]\nk = 2\n")
    write_toml_preserving(path, {"a": {"k": 1}, "b": {"k": 3}, "c": {"z": True}})
    text = path.read_text()
    assert "# keep me" in text
    assert "k = 3" in text
    assert "[c]" in text
    write_toml_preserving(path, {"a": {"k": 1}})
    assert "[b]" not in path.read_text()
