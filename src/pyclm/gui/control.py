"""
The control window (``pyclm control <dir>``): a Qt window in its own
process that prepares an experiment directory and drives a run from the
outside. It never shares a process with the run, so a crash here cannot
touch the results.

- **Run**: the check report, start / rehearse the run as a subprocess,
  pause, resume, stop, commands to the running experiment (files in
  ``commands/``), and the status line and minimap.
- **Positions**: the stage (pymmcore-widgets on the real microscope, a
  small simulated panel in ``--dry``), a position table with an experiment
  column, add / move to / preview here / save ``PositionList.pos``.
- **Files**: forms for the three configuration files, generated from the
  schema.

The microscope core is owned by this window only while no run is active
(§5 of docs/stage5b-interactivity-design.md).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
from qtpy import QtCore, QtWidgets

from ..check import CheckReport, check_directory, find_pyclm_config
from ..commands import write_command
from ..core.experiments import ImagingConfig, MicroscopePosition
from ..directories import positions_from_pos, positions_from_xml, write_position_list
from ..schema import ConfigError, ExperimentConfig, PyclmConfig
from .widgets import RunOverview

logger = logging.getLogger(__name__)


# ============================================================ stage access
class StageAccess(Protocol):
    """What the positions panel needs from a microscope."""

    xy_device: str
    z_device: str

    def position(self) -> tuple[float, float, float]: ...

    def move_to(self, x: float, y: float, z: float) -> None: ...

    def snap(self, cfg: ImagingConfig | None) -> np.ndarray: ...

    def pfs_offset(self) -> float | None: ...

    def release(self) -> None: ...


class SimulatedStage:
    """The positions panel on the virtual microscope (``--dry``)."""

    xy_device = "XYStage"
    z_device = "ZDrive"

    def __init__(self, core):
        self.core = core

    def position(self):
        x, y = self.core.getXYPosition()
        return float(x), float(y), float(self.core.getZPosition())

    def move_to(self, x, y, z):
        self.core.setXYPosition(x, y)
        self.core.setPosition(z)

    def snap(self, cfg):
        if cfg is not None:
            for g in cfg.get_config_groups():
                self.core.setConfig(g.group, g.config)
            self.core.setExposure(cfg.exposure)
        self.core.snapImage()
        return np.asarray(self.core.getImage())

    def pfs_offset(self):
        return None

    def release(self):
        return None


class MMCoreStage:
    """The positions panel on a real microscope through pymmcore-plus."""

    def __init__(self, core, pfs_device: str = "PFSOffset"):
        self.core = core
        self.xy_device = core.getXYStageDevice()
        self.z_device = core.getFocusDevice()
        self.pfs_device = pfs_device

    def position(self):
        x, y = self.core.getXYPosition()
        return float(x), float(y), float(self.core.getPosition())

    def move_to(self, x, y, z):
        self.core.setXYPosition(x, y)
        self.core.setPosition(z)
        self.core.waitForSystem()

    def snap(self, cfg):
        if cfg is not None:
            for g in cfg.get_config_groups():
                self.core.setConfig(g.group, g.config)
            for d in cfg.get_device_properties():
                self.core.setProperty(d.device, d.property, d.value)
            self.core.setExposure(cfg.exposure)
        self.core.waitForSystem()
        self.core.snapImage()
        return np.asarray(self.core.getImage())

    def pfs_offset(self):
        try:
            if self.pfs_device in self.core.getLoadedDevices():
                return float(self.core.getPosition(self.pfs_device))
        except Exception:
            return None
        return None

    def release(self):
        try:
            self.core.unloadAllDevices()
        except Exception:
            pass


# ================================================================ run panel
class RunPanel(QtWidgets.QWidget):
    """Check, start, watch and steer a run; every action is a file or a subprocess."""

    def __init__(self, window: ControlWindow):
        super().__init__(window)
        self.window = window
        self.process: QtCore.QProcess | None = None

        self.report = QtWidgets.QPlainTextEdit(readOnly=True)
        self.report.setMaximumBlockCount(2000)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(5000)
        self.overview = RunOverview(window.directory)

        self.check_button = QtWidgets.QPushButton("Check")
        self.rehearse_button = QtWidgets.QPushButton("Rehearse (dry)")
        self.start_button = QtWidgets.QPushButton("Start run")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.resume_button = QtWidgets.QPushButton("Resume")
        self.stop_button = QtWidgets.QPushButton("Stop run")
        self.check_button.clicked.connect(self.check)
        self.rehearse_button.clicked.connect(lambda: self.start(dry=True))
        self.start_button.clicked.connect(lambda: self.start(dry=False))
        self.pause_button.clicked.connect(lambda: self.send({"command": "pause"}))
        self.resume_button.clicked.connect(lambda: self.send({"command": "resume"}))
        self.stop_button.clicked.connect(lambda: self.send({"command": "stop_run"}))

        buttons = QtWidgets.QHBoxLayout()
        for b in (
            self.check_button,
            self.rehearse_button,
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
        ):
            buttons.addWidget(b)

        # commands to one experiment
        self.experiment = QtWidgets.QComboBox()
        self.channel = QtWidgets.QComboBox()
        self.exposure = QtWidgets.QDoubleSpinBox(
            minimum=0.01, maximum=100000, value=10, suffix=" ms"
        )
        self.exposure_button = QtWidgets.QPushButton("Set exposure")
        self.exposure_button.clicked.connect(self.send_exposure)
        self.device = QtWidgets.QLineEdit(placeholderText="device")
        self.prop = QtWidgets.QLineEdit(placeholderText="property")
        self.value = QtWidgets.QLineEdit(placeholderText="value")
        self.property_button = QtWidgets.QPushButton("Set property")
        self.property_button.clicked.connect(self.send_property)
        self.pos_x = QtWidgets.QDoubleSpinBox(minimum=-1e7, maximum=1e7, decimals=2)
        self.pos_y = QtWidgets.QDoubleSpinBox(minimum=-1e7, maximum=1e7, decimals=2)
        self.pos_z = QtWidgets.QDoubleSpinBox(minimum=-1e7, maximum=1e7, decimals=2)
        self.position_button = QtWidgets.QPushButton("Set position")
        self.position_button.clicked.connect(self.send_position)
        self.param_key = QtWidgets.QLineEdit(placeholderText="parameter")
        self.param_value = QtWidgets.QLineEdit(placeholderText="value")
        self.pattern_button = QtWidgets.QPushButton("Set pattern parameter")
        self.pattern_button.clicked.connect(self.send_pattern)
        self.stop_experiment_button = QtWidgets.QPushButton("Stop this experiment")
        self.stop_experiment_button.clicked.connect(
            lambda: self.send(
                {
                    "command": "stop_experiment",
                    "experiment": self.experiment.currentText(),
                }
            )
        )
        self.experiment.currentTextChanged.connect(self._fill_channels)

        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel("experiment"), 0, 0)
        form.addWidget(self.experiment, 0, 1)
        form.addWidget(self.stop_experiment_button, 0, 2)
        form.addWidget(QtWidgets.QLabel("channel"), 1, 0)
        form.addWidget(self.channel, 1, 1)
        form.addWidget(self.exposure, 2, 1)
        form.addWidget(self.exposure_button, 2, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.device)
        row.addWidget(self.prop)
        row.addWidget(self.value)
        form.addLayout(row, 3, 1)
        form.addWidget(self.property_button, 3, 2)
        row = QtWidgets.QHBoxLayout()
        for w in (self.pos_x, self.pos_y, self.pos_z):
            row.addWidget(w)
        form.addLayout(row, 4, 1)
        form.addWidget(self.position_button, 4, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.param_key)
        row.addWidget(self.param_value)
        form.addLayout(row, 5, 1)
        form.addWidget(self.pattern_button, 5, 2)
        commands = QtWidgets.QGroupBox("Commands to the running experiment")
        commands.setLayout(form)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(buttons)
        left.addWidget(QtWidgets.QLabel("Check report"))
        left.addWidget(self.report, stretch=1)
        left.addWidget(commands)
        left.addWidget(QtWidgets.QLabel("Run output"))
        left.addWidget(self.log, stretch=1)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addLayout(left, stretch=2)
        layout.addWidget(self.overview, stretch=1)
        self.set_directory(window.directory)

    # ----------------------------------------------------------- state
    @property
    def run_active(self) -> bool:
        return (
            self.process is not None
            and self.process.state() != QtCore.QProcess.NotRunning
        )

    def set_directory(self, directory: Path | None):
        self.overview.set_directory(directory) if directory else None
        self._fill_experiments()
        self.update_buttons()

    def _fill_experiments(self):
        self.experiment.clear()
        directory = self.window.directory
        if directory is None:
            return
        labels = []
        for name in ("PositionList.pos", "multipoints.xml"):
            path = directory / name
            if path.exists():
                try:
                    reader = (
                        positions_from_pos
                        if name.endswith(".pos")
                        else positions_from_xml
                    )
                    labels = [p.label for p in reader(str(path))]
                except Exception as e:
                    logger.warning(f"{name}: {e}")
                break
        self.experiment.addItems(labels)

    def _fill_channels(self, label: str):
        self.channel.clear()
        directory = self.window.directory
        if not label or directory is None:
            return
        toml = directory / f"{label.split('.')[0]}.toml"
        try:
            cfg = ExperimentConfig.from_file(toml)
        except ConfigError:
            return
        self.channel.addItems([*cfg.channels.presets, "stimulation"])
        self.exposure.setValue(cfg.imaging.exposure)

    def update_buttons(self):
        active = self.run_active
        self.start_button.setEnabled(not active and self.window.directory is not None)
        self.rehearse_button.setEnabled(
            not active and self.window.directory is not None
        )
        for b in (
            self.pause_button,
            self.resume_button,
            self.stop_button,
            self.stop_experiment_button,
            self.exposure_button,
            self.property_button,
            self.position_button,
            self.pattern_button,
        ):
            b.setEnabled(active)

    # --------------------------------------------------------- actions
    def check(self) -> CheckReport | None:
        directory = self.window.directory
        if directory is None:
            return None
        report = check_directory(directory, self.window.config_path)
        self.report.setPlainText(report.text())
        return report

    def start(self, dry: bool = False):
        directory = self.window.directory
        if directory is None or self.run_active:
            return
        report = self.check()
        if report is not None and not report.ok:
            self.log.appendPlainText("not started: the check found errors")
            return
        self.window.release_core()
        args = [sys.executable, "-m", "pyclm", "run", str(directory), "--no-check"]
        if dry:
            args.append("--dry")
        if self.window.config_path is not None:
            args += ["--config", str(self.window.config_path)]
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.log.appendPlainText("$ " + " ".join(args))
        self.process.start(args[0], args[1:])
        self.update_buttons()

    def _read_output(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            self.log.appendPlainText(line)

    def _finished(self, code, _status=None):
        self.log.appendPlainText(f"run finished with exit code {code}")
        self.update_buttons()
        self.window.run_finished()

    def send(self, command: dict) -> Path | None:
        directory = self.window.directory
        if directory is None:
            return None
        try:
            path = write_command(directory, command)
        except ValueError as e:
            self.log.appendPlainText(f"command refused: {e}")
            return None
        self.log.appendPlainText(f"command written: {path.name}  {command}")
        return path

    def send_exposure(self):
        self.send(
            {
                "command": "set_exposure",
                "experiment": self.experiment.currentText(),
                "channel": self.channel.currentText(),
                "ms": self.exposure.value(),
            }
        )

    def send_property(self):
        self.send(
            {
                "command": "set_property",
                "experiment": self.experiment.currentText(),
                "channel": self.channel.currentText(),
                "device": self.device.text().strip(),
                "property": self.prop.text().strip(),
                "value": _parse_value(self.value.text()),
            }
        )

    def send_position(self):
        self.send(
            {
                "command": "set_position",
                "experiment": self.experiment.currentText(),
                "x": self.pos_x.value(),
                "y": self.pos_y.value(),
                "z": self.pos_z.value(),
            }
        )

    def send_pattern(self):
        key = self.param_key.text().strip()
        if not key:
            return
        self.send(
            {
                "command": "set_pattern",
                "experiment": self.experiment.currentText(),
                "parameters": {key: _parse_value(self.param_value.text())},
            }
        )

    def refresh(self):
        self.overview.refresh()


def _parse_value(text: str):
    """A typed value from a line edit: bool, int, float, else the string."""
    t = text.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            continue
    return t


# ========================================================= positions panel
COLUMNS = ("label", "experiment", "x", "y", "z", "pfs_offset")


class PositionsPanel(QtWidgets.QWidget):
    """A position table with an experiment column, fed by the stage; saves PositionList.pos."""

    def __init__(self, window: ControlWindow):
        super().__init__(window)
        self.window = window
        self.stage: StageAccess | None = None
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.load_button = QtWidgets.QPushButton("Load list")
        self.add_button = QtWidgets.QPushButton("Add current position")
        self.move_button = QtWidgets.QPushButton("Move to selected")
        self.remove_button = QtWidgets.QPushButton("Remove selected")
        self.preview_button = QtWidgets.QPushButton("Preview here")
        self.save_button = QtWidgets.QPushButton("Save PositionList.pos")
        self.load_button.clicked.connect(self.load)
        self.add_button.clicked.connect(self.add_current)
        self.move_button.clicked.connect(self.move_to_selected)
        self.remove_button.clicked.connect(self.remove_selected)
        self.preview_button.clicked.connect(self.preview_here)
        self.save_button.clicked.connect(self.save)
        self.message = QtWidgets.QLabel("")
        self.message.setWordWrap(True)

        buttons = QtWidgets.QHBoxLayout()
        for b in (
            self.load_button,
            self.add_button,
            self.move_button,
            self.remove_button,
            self.preview_button,
            self.save_button,
        ):
            buttons.addWidget(b)
        self.hardware = QtWidgets.QWidget()
        self.hardware_layout = QtWidgets.QVBoxLayout(self.hardware)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.hardware)
        layout.addLayout(buttons)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.message)
        self.set_stage(None)

    # ----------------------------------------------------------- stage
    def set_stage(
        self, stage: StageAccess | None, widgets: list[QtWidgets.QWidget] | None = None
    ):
        self.stage = stage
        while self.hardware_layout.count():
            item = self.hardware_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for w in widgets or []:
            self.hardware_layout.addWidget(w)
        if stage is None:
            self.hardware_layout.addWidget(
                QtWidgets.QLabel(
                    "no microscope attached (a run is active, or open with --dry)"
                )
            )
        for b in (self.add_button, self.move_button, self.preview_button):
            b.setEnabled(stage is not None)

    # ------------------------------------------------------------ rows
    def experiments(self) -> list[str]:
        directory = self.window.directory
        if directory is None:
            return []
        return sorted(
            p.stem
            for p in directory.glob("*.toml")
            if p.name not in ("pyclm_config.toml", "schedule.toml")
        )

    def rows(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            row = {}
            for c, name in enumerate(COLUMNS):
                widget = self.table.cellWidget(r, c)
                if isinstance(widget, QtWidgets.QComboBox):
                    row[name] = widget.currentText()
                else:
                    item = self.table.item(r, c)
                    row[name] = item.text() if item is not None else ""
            out.append(row)
        return out

    def add_row(
        self, label: str, experiment: str, x: float, y: float, z: float, pfs=None
    ):
        r = self.table.rowCount()
        self.table.insertRow(r)
        combo = QtWidgets.QComboBox()
        combo.addItems(self.experiments())
        if experiment in self.experiments():
            combo.setCurrentText(experiment)
        combo.currentTextChanged.connect(lambda _t, r=r: self._relabel(r))
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
        self.table.setCellWidget(r, 1, combo)
        for c, value in zip((2, 3, 4), (x, y, z), strict=True):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(f"{value:.2f}"))
        self.table.setItem(
            r, 5, QtWidgets.QTableWidgetItem("" if pfs is None else f"{pfs:.2f}")
        )

    def _relabel(self, r: int):
        """Keep the label in the ``<experiment>.<n>`` form when the experiment changes."""
        combo = self.table.cellWidget(r, 1)
        item = self.table.item(r, 0)
        if combo is None or item is None:
            return
        stem = combo.currentText()
        suffix = item.text().split(".", 1)[1] if "." in item.text() else f"{r:02d}"
        item.setText(f"{stem}.{suffix}")

    def next_label(self, stem: str) -> str:
        used = {row["label"] for row in self.rows()}
        n = 0
        while f"{stem}.{n:02d}" in used:
            n += 1
        return f"{stem}.{n:02d}"

    def selected_row(self) -> int | None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        return min(rows) if rows else None

    # --------------------------------------------------------- actions
    def load(self):
        directory = self.window.directory
        if directory is None:
            return
        self.table.setRowCount(0)
        for name, reader in (
            ("PositionList.pos", positions_from_pos),
            ("multipoints.xml", positions_from_xml),
        ):
            path = directory / name
            if path.exists():
                for p in reader(str(path)):
                    extras = getattr(p, "extras", {}) or {}
                    pfs = extras.get("PFSOffset", getattr(p, "autofocus_offset", None))
                    self.add_row(p.label, p.label.split(".")[0], p.x, p.y, p.z, pfs)
                self.message.setText(
                    f"loaded {self.table.rowCount()} positions from {name}"
                )
                return
        self.message.setText("no position list in the directory")

    def add_current(self):
        if self.stage is None:
            return
        stems = self.experiments()
        if not stems:
            self.message.setText("no experiment files in the directory")
            return
        x, y, z = self.stage.position()
        stem = stems[0]
        r = self.selected_row()
        if r is not None:
            stem = self.table.cellWidget(r, 1).currentText()
        self.add_row(self.next_label(stem), stem, x, y, z, self.stage.pfs_offset())
        self.message.setText(
            f"added {self.rows()[-1]['label']} at x={x:.1f} y={y:.1f} z={z:.1f}"
        )

    def move_to_selected(self):
        r = self.selected_row()
        if r is None or self.stage is None:
            return
        row = self.rows()[r]
        self.stage.move_to(float(row["x"]), float(row["y"]), float(row["z"]))
        self.message.setText(f"moved to {row['label']}")

    def remove_selected(self):
        r = self.selected_row()
        if r is not None:
            self.table.removeRow(r)

    def preview_here(self):
        r = self.selected_row()
        directory = self.window.directory
        if r is None or self.stage is None or directory is None:
            return
        row = self.rows()[r]
        import tempfile

        import tifffile

        from ..preview import preview

        cfg = ExperimentConfig.from_file(
            directory / f"{row['experiment']}.toml"
        ).to_experiment(row["label"])
        frame = self.stage.snap(next(iter(cfg.channels.values())))
        tmp = Path(tempfile.mkdtemp(prefix="pyclm-preview-")) / "snap.tif"
        tifffile.imwrite(tmp, frame)
        try:
            result = preview(
                directory, row["label"], image=tmp, config_path=self.window.config_path
            )
            self.message.setText(result.text())
        except Exception as e:
            self.message.setText(f"preview failed: {e}")

    def save(self) -> Path | None:
        directory = self.window.directory
        if directory is None:
            return None
        rows = self.rows()
        positions = []
        for row in rows:
            extras = {}
            if row["pfs_offset"]:
                extras["PFSOffset"] = float(row["pfs_offset"])
            positions.append(
                MicroscopePosition(
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    label=row["label"],
                    extras=extras,
                )
            )
        xy = self.stage.xy_device if self.stage is not None else "XYStage"
        z = self.stage.z_device if self.stage is not None else "ZDrive"
        path = write_position_list(directory / "PositionList.pos", positions, xy, z)
        self.message.setText(f"saved {len(positions)} positions to {path.name}")
        self.window.run_panel._fill_experiments()
        return path


# ============================================================== the window
class ControlWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        directory: Path | None = None,
        dry: bool = False,
        config_path=None,
        attach_core: bool = True,
    ):
        super().__init__()
        self.setWindowTitle("PyCLM control")
        self.directory = None if directory is None else Path(directory)
        self.dry = dry
        self.config_path = None if config_path is None else Path(config_path)
        self._core = None

        self.tabs = QtWidgets.QTabWidget()
        self.run_panel = RunPanel(self)
        self.positions_panel = PositionsPanel(self)
        self.tabs.addTab(self.run_panel, "Run")
        self.tabs.addTab(self.positions_panel, "Positions")
        try:
            from .forms import FilesPanel

            self.files_panel = FilesPanel(self)
            self.tabs.addTab(self.files_panel, "Files")
        except ImportError:
            self.files_panel = None
        self.setCentralWidget(self.tabs)
        self.resize(1100, 750)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.run_panel.refresh)
        self._timer.start()
        if attach_core:
            self.attach_core()

    # ---------------------------------------------------------- the core
    def attach_core(self):
        """Own the microscope while no run is active: real through pymmcore-plus, or simulated."""
        if self.run_panel.run_active or self.directory is None:
            return
        try:
            if self.dry:
                from ..core.virtual_microscope.simulated_core import (
                    SimulatedMicroscopeCore,
                )
                from ..directories import dry_schedule_from_directory

                _schedule, source = dry_schedule_from_directory(self.directory)
                core = SimulatedMicroscopeCore(
                    source, pixel_size_um=source.pixel_size_um
                )
                self._core = core
                self.positions_panel.set_stage(
                    SimulatedStage(core), [QtWidgets.QLabel("virtual microscope")]
                )
                return
            cfg_path = find_pyclm_config(self.directory, self.config_path)
            if cfg_path is None:
                self.positions_panel.set_stage(None)
                return
            config = PyclmConfig.from_file(cfg_path)
            from pymmcore_plus import CMMCorePlus

            core = CMMCorePlus.instance()
            core.loadSystemConfiguration(config.config_path)
            if config.focus_device:
                core.setFocusDevice(config.focus_device)
            self._core = core
            widgets = self._hardware_widgets(core)
            self.positions_panel.set_stage(MMCoreStage(core), widgets)
        except Exception as e:
            logger.warning(f"microscope not attached: {e}")
            self.positions_panel.set_stage(None)
            self.positions_panel.message.setText(f"microscope not attached: {e}")

    def _hardware_widgets(self, core) -> list[QtWidgets.QWidget]:
        """pymmcore-widgets for stage, snap and live, when available."""
        try:
            from pymmcore_widgets import (
                ExposureWidget,
                ImagePreview,
                LiveButton,
                SnapButton,
                StageWidget,
            )
        except ImportError:
            return []
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        for device in (core.getXYStageDevice(), core.getFocusDevice()):
            if device:
                layout.addWidget(StageWidget(device=device, mmcore=core))
        buttons = QtWidgets.QVBoxLayout()
        buttons.addWidget(SnapButton(mmcore=core))
        buttons.addWidget(LiveButton(mmcore=core))
        buttons.addWidget(ExposureWidget(mmcore=core))
        layout.addLayout(buttons)
        layout.addWidget(ImagePreview(mmcore=core), stretch=1)
        return [row]

    def release_core(self):
        stage = self.positions_panel.stage
        if stage is not None:
            stage.release()
        self._core = None
        self.positions_panel.set_stage(None)

    def run_finished(self):
        self.attach_core()

    def set_directory(self, directory: Path):
        self.directory = Path(directory)
        self.run_panel.set_directory(self.directory)
        if self.files_panel is not None:
            self.files_panel.set_directory(self.directory)
        self.attach_core()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="PyCLM control window")
    p.add_argument("directory", nargs="?", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--dry", action="store_true", help="use the virtual microscope")
    args = p.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = ControlWindow(args.directory, dry=args.dry, config_path=args.config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
