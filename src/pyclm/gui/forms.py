"""
Forms for the three configuration files, generated from the schema
(:mod:`pyclm.schema`): every key's widget, limit and tooltip come from the
model, a method table shows the registered method names and the chosen
method's constructor arguments, and saving goes through the schema's own
validation and then ``tomlkit`` so comments in untouched tables survive.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import tomlkit
from pydantic import BaseModel
from qtpy import QtWidgets

from ..check import method_registries
from ..schema import (
    ConfigError,
    ExperimentConfig,
    MethodTable,
    PyclmConfig,
    ScheduleConfig,
    read_toml,
    validate_model,
)

logger = logging.getLogger(__name__)

RESERVED = {"pyclm_config.toml": PyclmConfig, "schedule.toml": ScheduleConfig}


def _unwrap_optional(annotation):
    args = get_args(annotation)
    if args and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        return inner[0] if len(inner) == 1 else annotation, True
    return annotation, False


def _limits(field) -> tuple[float | None, float | None]:
    lo = hi = None
    for meta in field.metadata:
        for attr in ("gt", "ge"):
            v = getattr(meta, attr, None)
            if v is not None:
                lo = v if attr == "ge" else v + 1e-9
        for attr in ("lt", "le"):
            v = getattr(meta, attr, None)
            if v is not None:
                hi = v if attr == "le" else v - 1e-9
    return lo, hi


def _parse_value(text: str):
    t = text.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            continue
    return t


class TableEditor(QtWidgets.QWidget):
    """A two-column key/value table (config groups, device properties)."""

    def __init__(self, values: dict | None = None, value_type=str, parent=None):
        super().__init__(parent)
        self.value_type = value_type
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(("key", "value"))
        self.table.horizontalHeader().setStretchLastSection(True)
        add = QtWidgets.QPushButton("+")
        remove = QtWidgets.QPushButton("-")
        add.clicked.connect(lambda: self.add_row("", ""))
        remove.clicked.connect(self.remove_selected)
        buttons = QtWidgets.QVBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table, stretch=1)
        layout.addLayout(buttons)
        self.table.setMaximumHeight(140)
        for k, v in (values or {}).items():
            self.add_row(str(k), str(v))

    def add_row(self, key: str, value: str):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(key))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(value))

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def value(self) -> dict:
        out = {}
        for r in range(self.table.rowCount()):
            k = self.table.item(r, 0)
            v = self.table.item(r, 1)
            key = k.text().strip() if k else ""
            if not key:
                continue
            text = v.text() if v else ""
            out[key] = text if self.value_type is str else _parse_value(text)
        return out


class ModelForm(QtWidgets.QWidget):
    """A form for one pydantic model; :meth:`value` returns a dict for validation."""

    def __init__(self, model: type[BaseModel], data: dict | None = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.widgets: dict[str, Any] = {}
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        data = dict(data or {})
        for name, field in model.model_fields.items():
            if name == "format_version":
                continue
            widget = self._widget_for(name, field, data.get(name))
            if widget is None:
                continue
            self.widgets[name] = widget
            label = QtWidgets.QLabel(name)
            if field.description:
                label.setToolTip(field.description)
                widget.setToolTip(field.description)
            layout.addRow(label, widget)

    def _widget_for(self, name: str, field, current):
        annotation, optional = _unwrap_optional(field.annotation)
        origin = get_origin(annotation)
        default = (
            None
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        value = current if current is not None else default
        lo, hi = _limits(field)

        if isinstance(annotation, type) and issubclass(annotation, MethodTable):
            return MethodForm(annotation, value, optional=optional)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            box = GroupForm(annotation, value, optional=optional, title=name)
            return box
        if origin is dict:
            _key_t, val_t = get_args(annotation)
            if isinstance(val_t, type) and issubclass(val_t, BaseModel):
                return NamedForms(val_t, value or {}, title=name)
            return TableEditor(value or {}, value_type=str if val_t is str else object)
        if origin is Literal:
            combo = QtWidgets.QComboBox()
            combo.addItems([str(a) for a in get_args(annotation)])
            if value is not None:
                combo.setCurrentText(str(value))
            return combo
        if origin is list:
            edit = QtWidgets.QLineEdit(", ".join(str(v) for v in (value or [])))
            edit.setPlaceholderText("comma separated")
            edit._is_list = True
            return edit
        if annotation is bool:
            box = QtWidgets.QCheckBox()
            box.setChecked(bool(value))
            return box
        if annotation is int:
            spin = QtWidgets.QSpinBox()
            spin.setRange(
                int(lo) if lo is not None else -1_000_000,
                int(hi) if hi is not None else 1_000_000,
            )
            spin.setValue(int(value) if value is not None else spin.minimum())
            return spin
        if annotation is float:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(
                float(lo) if lo is not None else -1e9,
                float(hi) if hi is not None else 1e9,
            )
            spin.setValue(float(value) if value is not None else spin.minimum())
            return spin
        if annotation is str or origin is None:
            return QtWidgets.QLineEdit("" if value is None else str(value))
        edit = QtWidgets.QLineEdit("" if value is None else str(value))
        return edit

    def value(self) -> dict:
        out = {}
        for name, widget in self.widgets.items():
            v = _value_of(widget)
            if v is None:
                continue
            out[name] = v
        return out


def _value_of(widget):
    if isinstance(widget, (GroupForm, MethodForm)):
        return widget.value()
    if isinstance(widget, NamedForms):
        return widget.value()
    if isinstance(widget, TableEditor):
        return widget.value()
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentText()
    if isinstance(widget, QtWidgets.QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QtWidgets.QSpinBox):
        return widget.value()
    if isinstance(widget, QtWidgets.QDoubleSpinBox):
        v = widget.value()
        return int(v) if float(v).is_integer() and widget.decimals() == 0 else v
    if isinstance(widget, QtWidgets.QLineEdit):
        text = widget.text().strip()
        if getattr(widget, "_is_list", False):
            return [t.strip() for t in text.split(",") if t.strip()]
        return text if text else None
    return None


class GroupForm(QtWidgets.QGroupBox):
    """A nested model as a group box; optional models get a checkbox."""

    def __init__(
        self, model: type[BaseModel], data, optional=False, title="", parent=None
    ):
        super().__init__(title, parent)
        self.optional = optional
        if optional:
            self.setCheckable(True)
            self.setChecked(data is not None)
        self.form = ModelForm(model, data if isinstance(data, dict) else _dump(data))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.form)

    def value(self):
        if self.optional and not self.isChecked():
            return None
        return self.form.value()


def _dump(data):
    if data is None:
        return {}
    if isinstance(data, BaseModel):
        out = data.model_dump()
        extra = getattr(data, "model_extra", None) or {}
        out.update(extra)
        return out
    return dict(data)


class NamedForms(QtWidgets.QGroupBox):
    """``dict[name, Model]`` tables ([channels.<preset>], [segmentation.<name>]) with add / remove."""

    def __init__(self, model: type[BaseModel], data: dict, title="", parent=None):
        super().__init__(title, parent)
        self.model = model
        self.entries: list[tuple[QtWidgets.QLineEdit, ModelForm | MethodForm]] = []
        self.rows = QtWidgets.QVBoxLayout()
        add = QtWidgets.QPushButton(f"add {title.rstrip('s')}")
        add.clicked.connect(lambda: self.add_entry("", {}))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(self.rows)
        layout.addWidget(add)
        for name, value in data.items():
            self.add_entry(name, _dump(value))

    def add_entry(self, name: str, data: dict):
        box = QtWidgets.QGroupBox()
        name_edit = QtWidgets.QLineEdit(name)
        name_edit.setPlaceholderText("name")
        form = (
            MethodForm(self.model, data)
            if issubclass(self.model, MethodTable)
            else ModelForm(self.model, data)
        )
        remove = QtWidgets.QPushButton("remove")
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("name"))
        head.addWidget(name_edit, stretch=1)
        head.addWidget(remove)
        inner = QtWidgets.QVBoxLayout(box)
        inner.addLayout(head)
        inner.addWidget(form)
        self.rows.addWidget(box)
        entry = (name_edit, form)
        self.entries.append(entry)

        def _remove():
            self.entries.remove(entry)
            box.setParent(None)
            box.deleteLater()

        remove.clicked.connect(_remove)

    def value(self) -> dict:
        out = {}
        for name_edit, form in self.entries:
            name = name_edit.text().strip()
            if name:
                out[name] = form.value()
        return out


class MethodForm(QtWidgets.QGroupBox):
    """A method table: the method from the registry, its own keys, and its constructor arguments."""

    def __init__(self, model: type[MethodTable], data, optional=False, parent=None):
        super().__init__(parent)
        self.model = model
        self.optional = optional
        data = _dump(data)
        if optional:
            self.setCheckable(True)
            self.setChecked(bool(data))
        kind = {"SegmentationTable": "segmentation", "TrackingTable": "tracking"}.get(
            model.__name__, "pattern"
        )
        patterns, segmentations, trackers = method_registries()
        self.registry = {
            "pattern": patterns,
            "segmentation": segmentations,
            "tracking": trackers,
        }[kind]
        self.kind = kind

        self.method = QtWidgets.QComboBox()
        self.method.setEditable(True)
        self.method.addItems(sorted(self.registry))
        self.method.setCurrentText(str(data.get("method", "")))
        self.fields: dict[str, Any] = {}
        layout = QtWidgets.QFormLayout(self)
        layout.addRow("method", self.method)
        for name, field in model.model_fields.items():
            if name == "method":
                continue
            annotation, _ = _unwrap_optional(field.annotation)
            value = data.get(name, field.default)
            if annotation is bool:
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(value))
            elif annotation is int:
                w = QtWidgets.QSpinBox()
                w.setRange(1, 1_000_000)
                w.setValue(int(value) if value is not None else 1)
            else:
                w = QtWidgets.QLineEdit("" if value is None else str(value))
            if field.description:
                w.setToolTip(field.description)
            self.fields[name] = w
            layout.addRow(name, w)
        own = set(model.model_fields)
        self.kwargs = TableEditor(
            {k: v for k, v in data.items() if k not in own}, value_type=object
        )
        layout.addRow("arguments", self.kwargs)
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        layout.addRow("", self.hint)
        self.method.currentTextChanged.connect(self._describe_method)
        self._describe_method(self.method.currentText())

    def _describe_method(self, name: str):
        cls = self.registry.get(name)
        if cls is None:
            self.hint.setText("not a registered method" if name else "")
            return
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            self.hint.setText("")
            return
        skip = (
            {"self", "experiment_name", "camera_properties", "channel"}
            if self.kind == "tracking"
            else {
                "self",
                "experiment_name",
                "camera_properties",
            }
        )
        params = [
            f"{p.name}={p.default!r}"
            if p.default is not inspect.Parameter.empty
            else p.name
            for p in sig.parameters.values()
            if p.name not in skip and p.kind is not inspect.Parameter.VAR_KEYWORD
        ]
        doc = (inspect.getdoc(cls) or "").split("\n")[0]
        self.hint.setText(
            (doc + "\n" if doc else "") + "arguments: " + (", ".join(params) or "none")
        )

    def value(self):
        if self.optional and not self.isChecked():
            return None
        out = {"method": self.method.currentText().strip()}
        for name, w in self.fields.items():
            if isinstance(w, QtWidgets.QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, QtWidgets.QSpinBox):
                out[name] = w.value()
            else:
                text = w.text().strip()
                if text:
                    out[name] = text
        out.update(self.kwargs.value())
        return out


# ------------------------------------------------------------- the panel
class FileForm(QtWidgets.QWidget):
    """One file: its form, its problems, save."""

    def __init__(self, path: Path, model: type[BaseModel], parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.model = model
        data = {}
        if self.path.exists():
            try:
                data = dict(read_toml(self.path))
            except ConfigError as e:
                data = {}
                logger.warning(str(e))
        data = _pre_form(model, data)
        self.form = ModelForm(model, data)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form)
        self.problems = QtWidgets.QLabel("")
        self.problems.setWordWrap(True)
        self.save_button = QtWidgets.QPushButton(f"Save {self.path.name}")
        self.save_button.clicked.connect(self.save)
        validate = QtWidgets.QPushButton("Validate")
        validate.clicked.connect(self.validate)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(validate)
        buttons.addWidget(self.save_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(self.problems)
        layout.addLayout(buttons)

    def document_value(self) -> dict:
        return _post_form(self.model, self.form.value())

    def validate(self) -> list[str]:
        try:
            validate_model(self.model, self.document_value(), self.path)
        except ConfigError as e:
            self.problems.setText("\n".join(e.problems))
            return e.problems
        self.problems.setText("ok")
        return []

    def save(self) -> bool:
        if self.validate():
            return False
        write_toml_preserving(self.path, self.document_value())
        self.problems.setText(f"saved {self.path.name}")
        return True


def _pre_form(model, data: dict) -> dict:
    """The file's dict in the shape the models use (named tables split out)."""
    data = dict(data)
    if model is ExperimentConfig:
        seg = data.get("segmentation")
        if isinstance(seg, dict):
            seg = dict(seg)
            named = {k: seg.pop(k) for k in list(seg) if isinstance(seg[k], dict)}
            data["segmentation"] = seg or None
            data["segmentations"] = named
        channels = data.get("channels")
        if isinstance(channels, dict):
            channels = dict(channels)
            overrides = {
                k: channels.pop(k)
                for k in list(channels)
                if isinstance(channels[k], dict)
            }
            channels["overrides"] = overrides
            data["channels"] = channels
    return data


def _post_form(model, value: dict) -> dict:
    """Back to the file's shape: named tables nested where the file expects them."""
    value = dict(value)
    if model is ExperimentConfig:
        named = value.pop("segmentations", {}) or {}
        seg = value.pop("segmentation", None)
        if seg or named:
            table = dict(seg or {})
            table.update(named)
            value["segmentation"] = table
        channels = dict(value.get("channels") or {})
        overrides = channels.pop("overrides", {}) or {}
        channels.update(overrides)
        value["channels"] = channels
        if value.get("tracking") is None:
            value.pop("tracking", None)
    # top-level keys first so t_delay / t_stop land above the first table
    ordered = {k: v for k, v in value.items() if not isinstance(v, dict)}
    ordered.update({k: v for k, v in value.items() if isinstance(v, dict)})
    return ordered


def write_toml_preserving(path: Path, value: dict) -> None:
    """
    Write ``value`` into the file, changing only what changed: existing
    tables are updated in place (their comments and layout survive), keys
    that are gone are removed, new keys are added (tomlkit).
    """
    path = Path(path)
    if path.exists():
        doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    else:
        doc = tomlkit.document()
    _merge(doc, value)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _merge(container, new: dict) -> None:
    for key in list(container.keys()):
        if key not in new:
            del container[key]
    for key, value in new.items():
        old = container.get(key)
        if isinstance(value, dict) and old is not None and hasattr(old, "keys"):
            _merge(old, value)
        elif old is not None and _plain(old) == value:
            continue
        else:
            container[key] = value


def _plain(item):
    try:
        return item.unwrap()
    except AttributeError:
        return item


class FilesPanel(QtWidgets.QWidget):
    """One tab per configuration file in the directory."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.tabs = QtWidgets.QTabWidget()
        self.new_name = QtWidgets.QLineEdit(placeholderText="new experiment name")
        self.new_button = QtWidgets.QPushButton("New experiment file")
        self.new_button.clicked.connect(self.new_experiment)
        self.reload_button = QtWidgets.QPushButton("Reload")
        self.reload_button.clicked.connect(
            lambda: self.set_directory(self.window.directory)
        )
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.new_name)
        row.addWidget(self.new_button)
        row.addWidget(self.reload_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.tabs, stretch=1)
        self.forms: dict[str, FileForm] = {}
        self.set_directory(window.directory)

    def set_directory(self, directory):
        self.tabs.clear()
        self.forms = {}
        if directory is None:
            return
        directory = Path(directory)
        files = [directory / "pyclm_config.toml", directory / "schedule.toml"]
        files += sorted(p for p in directory.glob("*.toml") if p.name not in RESERVED)
        for path in files:
            model = RESERVED.get(path.name, ExperimentConfig)
            form = FileForm(path, model)
            self.forms[path.name] = form
            self.tabs.addTab(form, path.name)

    def new_experiment(self):
        name = self.new_name.text().strip()
        directory = self.window.directory
        if not name or directory is None:
            return
        from ..templates import TEMPLATES

        path = directory / f"{name}.toml"
        if not path.exists():
            path.write_text(
                TEMPLATES["closed-loop"].format(name=name), encoding="utf-8"
            )
        self.set_directory(directory)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)
