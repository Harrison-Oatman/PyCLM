"""Custom methods from pyclm_config.toml `methods` and the pyclm.methods entry points."""

import shutil
import types

import numpy as np
import pytest
from test_check import RESOURCES, TOMLS
from test_dry_run import yml_experiment_dir

from pyclm import PatternMethod
from pyclm import methods as pm
from pyclm.check import check_directory
from pyclm.methods import MethodLoadError, discover, load_methods

HALF_ON = '''
import numpy as np
from pyclm import PatternMethod


class _Base(PatternMethod):
    """A shared base without a name: not registered."""

    def level(self):
        return 1.0


class HalfOn(_Base):
    name = "half_on"

    def __init__(self, side="left", **kwargs):
        super().__init__(**kwargs)
        self.side = side

    def generate(self, context):
        out = np.zeros(self.pattern_shape, np.float32)
        w = out.shape[1] // 2
        if self.side == "left":
            out[:, :w] = self.level()
        else:
            out[:, w:] = self.level()
        return out
'''


def _with_methods(directory, *specs):
    config = directory / "pyclm_config.toml"
    text = config.read_text()
    listed = ", ".join(f'"{s}"' for s in specs)
    # top-level key: before the first table
    head, sep, tail = text.partition("\n[")
    config.write_text(f"{head}\nmethods = [{listed}]\n{sep}{tail}")


def _use_method(toml, method_block):
    text = toml.read_text()
    toml.write_text(text[: text.index("[pattern]")] + method_block)


# ------------------------------------------------------------------ loading
def test_a_file_registers_the_classes_it_defines(tmp_path):
    (tmp_path / "my_patterns.py").write_text(HALF_ON)
    found = load_methods(["my_patterns.py"], tmp_path)
    assert sorted(found.pattern) == ["half_on"]  # not _Base, not PatternMethod
    assert issubclass(found.pattern["half_on"], PatternMethod)
    assert found.summary() == ["pattern method 'half_on' from my_patterns.py"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "from pyclm import PatternMethod\nclass Nameless(PatternMethod):\n    pass\n",
            "without its own name",
        ),
        (
            "from pyclm import PatternMethod\nclass Bar(PatternMethod):\n    name = 'bar'\n",
            "name of a built-in pattern method",
        ),
        ("x = 1\n", "defines no PatternMethod"),
        ("import not_a_module_anywhere\n", "ModuleNotFoundError"),
    ],
)
def test_problems_are_named(tmp_path, source, message):
    (tmp_path / "bad.py").write_text(source)
    with pytest.raises(MethodLoadError, match=message):
        load_methods(["bad.py"], tmp_path)


def test_one_name_in_two_files_is_an_error(tmp_path):
    (tmp_path / "a.py").write_text(HALF_ON)
    (tmp_path / "b.py").write_text(HALF_ON)
    with pytest.raises(MethodLoadError, match="defined twice"):
        load_methods(["a.py", "b.py"], tmp_path)


def test_missing_file(tmp_path):
    with pytest.raises(MethodLoadError, match="not found"):
        load_methods(["nope.py"], tmp_path)


def test_entry_points(monkeypatch, tmp_path):
    module = types.ModuleType("mylab_patterns")

    class Spot(PatternMethod):
        name = "spot"

        def generate(self, context):
            return np.zeros(self.pattern_shape, np.float32)

    Spot.__module__ = module.__name__
    module.Spot = Spot

    class EP:
        def __init__(self, name, value, target):
            self.name, self.value, self._target = name, value, target

        def load(self):
            if isinstance(self._target, Exception):
                raise self._target
            return self._target

    eps = [
        EP("mylab", "mylab_patterns", module),
        EP("broken", "broken_pkg", ImportError("no module named broken_pkg")),
    ]
    monkeypatch.setattr(pm, "entry_points", lambda group: eps)
    found, warnings = discover()
    assert found.pattern == {"spot": Spot}
    assert len(warnings) == 1
    assert "broken" in warnings[0]


# ------------------------------------------------------------------ check, run
def _check_dir(tmp_path):
    for name in ("pyclm_config.toml", "PositionList.pos"):
        shutil.copy(TOMLS / name, tmp_path / name)
    shutil.copy(RESOURCES / "test_schedule.toml", tmp_path / "schedule.toml")
    for stem in ("bar10", "bar025"):
        shutil.copy(TOMLS / f"{stem}.toml", tmp_path / f"{stem}.toml")
    return tmp_path


def test_check_loads_methods_from_the_config(tmp_path):
    d = _check_dir(tmp_path)
    _use_method(d / "bar10.toml", '[pattern]\nmethod = "half_on"\nside = "right"\n')
    report = check_directory(d)
    assert "unknown pattern method 'half_on'" in report.text()
    assert "methods in pyclm_config.toml" in report.text()  # the hint

    (d / "my_patterns.py").write_text(HALF_ON)
    _with_methods(d, "my_patterns.py")
    report = check_directory(d)
    assert report.ok, report.text()
    assert "pattern method 'half_on' from my_patterns.py" in report.text()

    _use_method(d / "bar10.toml", '[pattern]\nmethod = "half_on"\nsid = "right"\n')
    assert "unknown argument 'sid'" in check_directory(d).text()


def test_check_reports_a_broken_methods_file(tmp_path):
    d = _check_dir(tmp_path)
    (d / "my_patterns.py").write_text("def broken(:\n")
    _with_methods(d, "my_patterns.py")
    report = check_directory(d)
    assert not report.ok
    assert "SyntaxError" in report.text()


def test_dry_run_with_a_config_method(yml_experiment_dir):
    import pyclm.io as pio
    from pyclm import run_pyclm

    d = yml_experiment_dir
    config = d / "pyclm_config.toml"
    config.write_text(
        config.read_text().replace('format = "hdf5"', 'format = "ome-zarr"')
    )
    (d / "my_patterns.py").write_text(HALF_ON)
    _with_methods(d, "my_patterns.py")
    _use_method(d / "bar10.toml", '[pattern]\nmethod = "half_on"\nside = "right"\n')
    run_pyclm(d, dry=True)

    with pio.open(d / "bar10.00.zarr") as exp:
        cam = exp.camera_pattern_at(exp.current_t)
    _h, w = cam.shape
    assert cam[:, : w // 2].max() == 0
    assert cam[:, w // 2 + 1 :].min() == 255


def test_methods_given_in_code_win(tmp_path):
    (tmp_path / "my_patterns.py").write_text(HALF_ON)
    found = load_methods(["my_patterns.py"], tmp_path)

    class Other(PatternMethod):
        name = "half_on"

    patterns, _, _ = found.merged_with(pattern={"half_on": Other})
    assert patterns["half_on"] is Other
