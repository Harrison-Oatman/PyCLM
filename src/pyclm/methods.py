"""
Custom methods without writing Python to run them.

Pattern, segmentation and tracking methods outside PyCLM reach a run in
two ways:

- ``methods`` in ``pyclm_config.toml``: a list of Python files (relative
  to the configuration file) or importable module names::

      methods = ["my_patterns.py", "mylab.patterns"]

- the ``pyclm.methods`` entry-point group, for an installed package::

      [project.entry-points."pyclm.methods"]
      mylab = "mylab.patterns"

Every :class:`~pyclm.PatternMethod`, :class:`~pyclm.SegmentationMethod`
and :class:`~pyclm.TrackingMethod` subclass *defined* in such a module is
registered under its ``name`` class attribute, which is what
``[pattern] method = ...`` (or ``[segmentation]``, ``[tracking]``) selects.
Classes the module imports from elsewhere are not registered again.
``pyclm run``, ``pyclm check`` and ``pyclm preview`` all load them; methods
passed to :func:`~pyclm.run_pyclm` in code take precedence.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "pyclm.methods"


class MethodLoadError(Exception):
    """A methods module could not be loaded or holds a method PyCLM cannot register."""


@dataclass
class MethodSet:
    """Custom methods by kind, as ``{name: class}``, and where each came from."""

    pattern: dict[str, type] = field(default_factory=dict)
    segmentation: dict[str, type] = field(default_factory=dict)
    tracking: dict[str, type] = field(default_factory=dict)
    sources: dict[tuple[str, str], str] = field(default_factory=dict)

    def kinds(self) -> dict[str, dict[str, type]]:
        return {
            "pattern": self.pattern,
            "segmentation": self.segmentation,
            "tracking": self.tracking,
        }

    def __len__(self) -> int:
        return len(self.pattern) + len(self.segmentation) + len(self.tracking)

    def add(self, kind: str, name: str, cls: type, source: str) -> None:
        table = self.kinds()[kind]
        have = table.get(name)
        if have is not None and have is not cls:
            raise MethodLoadError(
                f"{kind} method {name!r} is defined twice: in "
                f"{self.sources[(kind, name)]} and in {source}"
            )
        table[name] = cls
        self.sources[(kind, name)] = source

    def merged_with(
        self, pattern=None, segmentation=None, tracking=None
    ) -> tuple[dict, dict, dict]:
        """These methods plus ones given in code (which win), per kind."""
        return (
            {**self.pattern, **(pattern or {})},
            {**self.segmentation, **(segmentation or {})},
            {**self.tracking, **(tracking or {})},
        )

    def summary(self) -> list[str]:
        return [
            f"{kind} method {name!r} from {self.sources[(kind, name)]}"
            for kind, table in self.kinds().items()
            for name in sorted(table)
        ]


def _bases() -> dict[str, type]:
    from .core.patterns.pattern import PatternMethod
    from .core.segmentation.segmentation import SegmentationMethod
    from .core.tracking.tracking import TrackingMethod

    return {
        "pattern": PatternMethod,
        "segmentation": SegmentationMethod,
        "tracking": TrackingMethod,
    }


def _builtin_names() -> dict[str, set[str]]:
    from .core.patterns import known_models
    from .core.segmentation_process import SegmentationProcess
    from .core.tracking import known_tracking_methods

    return {
        "pattern": set(known_models),
        "segmentation": set(SegmentationProcess.default_models),
        "tracking": set(known_tracking_methods),
    }


def methods_in_module(module: ModuleType, source: str, into: MethodSet) -> int:
    """Register the method classes ``module`` defines; returns how many."""
    bases = _bases()
    builtins = _builtin_names()
    found = 0
    defined = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__ and not inspect.isabstract(cls)
    ]
    # a class another one here builds on may be a shared base without a name
    parents = {base for cls in defined for base in cls.__mro__[1:]}
    for cls in defined:
        for kind, base in bases.items():
            if not issubclass(cls, base) or cls is base:
                continue
            name = vars(cls).get("name")
            if (not isinstance(name, str) or not name) and cls in parents:
                continue
            if not isinstance(name, str) or not name:
                raise MethodLoadError(
                    f"{cls.__name__} in {source} is a {kind} method without its "
                    f'own name; add  name = "..."  to the class (the TOML selects '
                    "methods by it)"
                )
            if name in builtins[kind]:
                raise MethodLoadError(
                    f"{cls.__name__} in {source}: {name!r} is the name of a built-in "
                    f"{kind} method; choose another name"
                )
            into.add(kind, name, cls, source)
            found += 1
    return found


def _import_file(path: Path) -> ModuleType:
    module_name = f"pyclm_user_methods.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MethodLoadError(f"{path} cannot be imported as a Python file")
    module = importlib.util.module_from_spec(spec)
    # siblings of the file (helper modules) import as they would from a script
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise MethodLoadError(f"{path}: {type(e).__name__}: {e}") from e
    return module


def load_methods(specs: list[str], base_dir: Path) -> MethodSet:
    """
    The methods of ``specs``: ``.py`` paths (relative to ``base_dir``) or
    module names. Raises :class:`MethodLoadError` on the first problem.
    """
    out = MethodSet()
    for spec in specs:
        if spec.endswith(".py") or "/" in spec or "\\" in spec:
            path = Path(spec)
            if not path.is_absolute():
                path = Path(base_dir) / path
            if not path.exists():
                raise MethodLoadError(f"methods file {spec!r} not found ({path})")
            module = _import_file(path.resolve())
            source = str(spec)
        else:
            try:
                module = importlib.import_module(spec)
            except Exception as e:
                raise MethodLoadError(
                    f"methods module {spec!r}: {type(e).__name__}: {e}"
                ) from e
            source = spec
        if methods_in_module(module, source, out) == 0:
            raise MethodLoadError(
                f"{source} defines no PatternMethod, SegmentationMethod or "
                "TrackingMethod subclass"
            )
    return out


def entry_point_methods(into: MethodSet | None = None) -> tuple[MethodSet, list[str]]:
    """
    The methods of installed packages' ``pyclm.methods`` entry points, and a
    warning per entry point that failed (an unrelated broken package must
    not stop a run).
    """
    out = into if into is not None else MethodSet()
    warnings = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        source = f"entry point {ep.name!r} ({ep.value})"
        try:
            target = ep.load()
            if isinstance(target, ModuleType):
                methods_in_module(target, source, out)
            elif inspect.isclass(target):
                module = sys.modules[target.__module__]
                before = len(out)
                methods_in_module(module, source, out)
                if len(out) == before:
                    raise MethodLoadError(f"{target.__name__} is not a method class")
            else:
                raise MethodLoadError("must name a module or a method class")
        except Exception as e:
            warnings.append(f"{source} not loaded: {e}")
    return out, warnings


def discover(
    config=None, config_path: Path | None = None
) -> tuple[MethodSet, list[str]]:
    """
    Every custom method a run sees: the entry points, then the
    configuration's ``methods``. Returns the methods and the entry-point
    warnings; a problem with ``methods`` raises :class:`MethodLoadError`.
    """
    methods, warnings = entry_point_methods()
    specs = list(getattr(config, "methods", None) or [])
    if specs:
        base = Path(config_path).parent if config_path is not None else Path.cwd()
        mine = load_methods(specs, base)
        for kind, table in mine.kinds().items():
            for name, cls in table.items():
                methods.add(kind, name, cls, mine.sources[(kind, name)])
    for line in methods.summary():
        logger.info(f"custom {line}")
    return methods, warnings
