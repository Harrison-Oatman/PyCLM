"""
Pattern methods built from other pattern methods.

:class:`SplitPattern` (``method = "split"``) runs one method on each half of
the field, the halves being ``left`` / ``right`` or ``top`` / ``bottom``
sub-tables of ``[pattern]``::

    [pattern]
    method = "split"
    boundary = 0.5          # where the halves meet, as a fraction of the width (or height)
    feather = 0             # pixels of linear blend across the boundary

    [pattern.left]
    method = "move_in"
    channel = "545"

    [pattern.right]
    method = "move_out"
    channel = "545"

Each half's method is an ordinary registered pattern method (built in or
custom) with its own arguments; it sees the whole field and the composite
keeps its pattern on its side of the boundary. Requirements are the union
of the halves', so the frames and segmentations each half asks for arrive
once and are shared.

:class:`WrappingPattern` is the base of the methods that wrap one inner
method given as a ``[pattern.inner]`` table (``pulsed``, ``recenter``,
``drift_correct``, ``auto_exposure``, ``settings_schedule``)::

    [pattern]
    method = "pulsed"
    period_min = 30
    on_min = 10

    [pattern.inner]
    method = "circle"
    radius_um = 100

The wrapper changes the inner method's pattern (``pulsed`` gates it) or
leaves it as it is and acts on the run (``recenter`` moves the stage,
``auto_exposure`` changes an exposure). Wrappers nest: the inner method may
itself be a wrapper or a ``split``.
"""

from __future__ import annotations

import logging

import numpy as np

from ..experiments import Experiment
from .pattern import (
    AcquiredImageRequest,
    CameraProperties,
    PatternMethod,
    merge_requests,
)
from .zoo import ZooMeta

logger = logging.getLogger(__name__)

HORIZONTAL = ("left", "right")  # the boundary is vertical: halves side by side
VERTICAL = ("top", "bottom")  # the boundary is horizontal: halves one above the other
REGIONS = HORIZONTAL + VERTICAL


class SplitPattern(PatternMethod):
    """One pattern method per half of the field; see the module docstring."""

    name = "split"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "boundary": 0.5,
            "left": {"method": "move_in", "channel": "638", "gradient": True},
            "right": {"method": "move_out", "channel": "638", "gradient": True},
        },
        title="Split Field",
        description="Two methods side by side (left / right) or one above the other (top / bottom), each in its own sub-table.",
        group="composite",
    )

    def __init__(
        self,
        left=None,
        right=None,
        top=None,
        bottom=None,
        boundary: float = 0.5,
        feather: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        specs = {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
        }
        given = tuple(r for r in REGIONS if specs[r] is not None)
        if given == HORIZONTAL:
            self.axis = 1
        elif given == VERTICAL:
            self.axis = 0
        else:
            raise ValueError(
                "split needs exactly [pattern.left] and [pattern.right], or "
                f"[pattern.top] and [pattern.bottom]; got {list(given) or 'none'}"
            )
        self.regions = given
        for region in given:
            spec = specs[region]
            if not isinstance(spec, dict) or "method" not in spec:
                raise ValueError(
                    f"[pattern.{region}] must be a table with a 'method' key"
                )
        self.specs = {r: dict(specs[r]) for r in given}
        if not 0.0 < float(boundary) < 1.0:
            raise ValueError(f"boundary must be between 0 and 1, got {boundary}")
        self.boundary = float(boundary)
        self.feather = max(0, int(feather))
        # built by bind_registry, which the pattern process calls before initialize
        self.children: dict[str, PatternMethod] = {}

    # ------------------------------------------------------------ construction
    @classmethod
    def nested_methods(cls, kwargs: dict) -> list[tuple[str, str, dict]]:
        out = []
        for region in REGIONS:
            spec = kwargs.get(region)
            if isinstance(spec, dict) and "method" in spec:
                rest = {k: v for k, v in spec.items() if k != "method"}
                out.append((region, str(spec["method"]), rest))
        return out

    def bind_registry(self, registry: dict[str, type]) -> None:
        for region, spec in self.specs.items():
            method = spec["method"]
            cls = registry.get(method)
            if cls is None:
                raise ValueError(
                    f"[pattern.{region}]: unknown pattern method {method!r} "
                    f"(registered: {', '.join(sorted(registry))})"
                )
            child_kwargs = {k: v for k, v in spec.items() if k != "method"}
            child = cls(**child_kwargs)
            child.bind_registry(registry)
            self.children[region] = child
        # mirror the halves' declared needs so tools that read them off the
        # method (pyclm check) see the union
        self._requirements_list = [
            req for c in self.children.values() for req in c._requirements_list
        ]
        stim = [c for c in self.children.values() if c._stim_requested]
        self._stim_requested = bool(stim)
        self._stim_request_raw = any(c._stim_request_raw for c in stim)
        self._stim_request_seg = any(c._stim_request_seg for c in stim)
        self._stim_request_history = max(
            (c._stim_request_history for c in stim), default=1
        )

    def _require_children(self) -> None:
        if len(self.children) != len(self.regions):
            raise RuntimeError(
                "split's halves are not built: the pattern process binds the "
                "method registry before initialize (bind_registry)"
            )

    # ------------------------------------------------------------ lifecycle
    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        self._require_children()
        return merge_requests(
            req
            for child in self.children.values()
            for req in child.initialize(experiment)
        )

    def configure_system(
        self,
        experiment_name: str,
        camera_properties: CameraProperties,
        experiment: Experiment,
    ):
        super().configure_system(experiment_name, camera_properties, experiment)
        for child in self.children.values():
            child.configure_system(experiment_name, camera_properties, experiment)

    def update_binning(self, binning: int):
        super().update_binning(binning)
        for child in self.children.values():
            if child.binning != int(binning):
                child.update_binning(binning)

    def update(self, **parameters) -> tuple[dict, dict]:
        """``boundary`` / ``feather`` here; ``left.<name>`` and the like go to that half."""
        own, forwarded = {}, {}
        for key, value in parameters.items():
            region, dot, rest = key.partition(".")
            if dot and region in self.children:
                forwarded.setdefault(region, {})[rest] = value
            else:
                own[key] = value
        applied, refused = super().update(**own)
        for region, params in forwarded.items():
            a, r = self.children[region].update(**params)
            applied.update({f"{region}.{k}": v for k, v in a.items()})
            refused.update({f"{region}.{k}": v for k, v in r.items()})
        return applied, refused

    # ------------------------------------------------------------ the pattern
    def masks(self) -> dict[str, np.ndarray]:
        """The weight of each half at every pixel (they sum to 1)."""
        h, w = self.pattern_shape
        n = w if self.axis == 1 else h
        coords = np.arange(n, dtype=np.float32) + 0.5
        edge = self.boundary * n
        if self.feather > 0:
            first = np.clip((edge - coords) / self.feather + 0.5, 0.0, 1.0)
        else:
            first = (coords < edge).astype(np.float32)
        first = first[np.newaxis, :] if self.axis == 1 else first[:, np.newaxis]
        first = np.broadcast_to(first, (h, w)).astype(np.float32)
        return {self.regions[0]: first, self.regions[1]: 1.0 - first}

    def generate(self, context) -> np.ndarray:
        self._require_children()
        masks = self.masks()
        out = np.zeros(self.pattern_shape, np.float32)
        for region, child in self.children.items():
            pattern = np.asarray(child.render(context), dtype=np.float32)
            if pattern.shape != out.shape:
                pattern = _fit(pattern, out.shape)
            out += masks[region] * np.clip(pattern, 0.0, 1.0)
        return np.clip(out, 0.0, 1.0)


def _fit(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Crop or zero-pad a half's pattern to the composite's shape."""
    out = np.zeros(shape, np.float32)
    h = min(shape[0], pattern.shape[0])
    w = min(shape[1], pattern.shape[1])
    out[:h, :w] = pattern[:h, :w]
    return out


class WrappingPattern(PatternMethod):
    """
    Base of the methods that wrap one inner method (``[pattern.inner]``);
    see the module docstring. A subclass implements :meth:`generate` with
    :meth:`inner_pattern`, and declares its own needs with
    ``add_requirement`` as any method does; the requests are the union of
    its own and the inner method's.
    """

    name = "wrapping_base"
    # whether [pattern.inner] must be given; without it the inner pattern is dark
    inner_required = True

    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        if inner is None:
            if self.inner_required:
                raise ValueError(
                    f"{self.name} wraps another method: add a [pattern.inner] "
                    'table with its method, e.g. method = "circle"'
                )
            self.spec = None
        elif not isinstance(inner, dict) or "method" not in inner:
            raise ValueError("[pattern.inner] must be a table with a 'method' key")
        else:
            self.spec = dict(inner)
        # built by bind_registry, which the pattern process calls before initialize
        self.children: dict[str, PatternMethod] = {}
        self._bound = False

    # ------------------------------------------------------------ construction
    @classmethod
    def nested_methods(cls, kwargs: dict) -> list[tuple[str, str, dict]]:
        spec = kwargs.get("inner")
        if isinstance(spec, dict) and "method" in spec:
            rest = {k: v for k, v in spec.items() if k != "method"}
            return [("inner", str(spec["method"]), rest)]
        return []

    def bind_registry(self, registry: dict[str, type]) -> None:
        if self._bound:
            return
        self._own_requirements = list(self._requirements_list)
        self._own_stim = self._stim_request()
        if self.spec is not None:
            method = self.spec["method"]
            cls = registry.get(method)
            if cls is None:
                raise ValueError(
                    f"[pattern.inner]: unknown pattern method {method!r} "
                    f"(registered: {', '.join(sorted(registry))})"
                )
            child = cls(**{k: v for k, v in self.spec.items() if k != "method"})
            child.bind_registry(registry)
            self.children["inner"] = child
            # mirror the inner method's needs, so tools that read them off the
            # method (pyclm check, the zoo) see the union
            self._requirements_list = self._own_requirements + list(
                child._requirements_list
            )
            if child._stim_requested:
                own = self._own_stim or (False, False, 1)
                self._stim_requested = True
                self._stim_request_raw = own[0] or child._stim_request_raw
                self._stim_request_seg = own[1] or child._stim_request_seg
                self._stim_request_history = max(own[2], child._stim_request_history)
        self._bound = True

    @property
    def inner(self) -> PatternMethod | None:
        return self.children.get("inner")

    def _require_inner(self) -> None:
        if self.spec is not None and self.inner is None:
            raise RuntimeError(
                f"{self.name}'s inner method is not built: the pattern process "
                "binds the method registry before initialize (bind_registry)"
            )

    # ------------------------------------------------------------ lifecycle
    def initialize(self, experiment: Experiment) -> list[AcquiredImageRequest]:
        self._require_inner()
        own_reqs = getattr(self, "_own_requirements", self._requirements_list)
        own_stim = getattr(self, "_own_stim", self._stim_request())
        reqs = self._build_requests(experiment, own_reqs, own_stim)
        if self.inner is not None:
            reqs += self.inner.initialize(experiment)
        return merge_requests(reqs)

    def configure_system(
        self,
        experiment_name: str,
        camera_properties: CameraProperties,
        experiment: Experiment,
    ):
        super().configure_system(experiment_name, camera_properties, experiment)
        if self.inner is not None:
            self.inner.configure_system(experiment_name, camera_properties, experiment)

    def update_binning(self, binning: int):
        super().update_binning(binning)
        inner = self.inner
        if inner is not None and inner.binning != int(binning):
            inner.update_binning(binning)

    def update(self, **parameters) -> tuple[dict, dict]:
        """Own parameters here; ``inner.<name>`` goes to the inner method."""
        own, forwarded = {}, {}
        for key, value in parameters.items():
            head, dot, rest = key.partition(".")
            if dot and head == "inner" and self.inner is not None:
                forwarded[rest] = value
            elif key == "inner":
                continue
            else:
                own[key] = value
        applied, refused = super().update(**own)
        if "inner" in parameters:
            refused["inner"] = "the inner method cannot be replaced during a run"
        if forwarded:
            a, r = self.inner.update(**forwarded)
            applied.update({f"inner.{k}": v for k, v in a.items()})
            refused.update({f"inner.{k}": v for k, v in r.items()})
        return applied, refused

    # ------------------------------------------------------------ the pattern
    def inner_pattern(self, context) -> np.ndarray:
        """The inner method's pattern (a dark field without one), as float32 in 0-1."""
        self._require_inner()
        if self.inner is None:
            return np.zeros(self.pattern_shape, np.float32)
        pattern = np.asarray(self.inner.render(context), dtype=np.float32)
        if pattern.shape != tuple(self.pattern_shape):
            pattern = _fit(pattern, tuple(int(v) for v in self.pattern_shape))
        return np.clip(pattern, 0.0, 1.0)

    def generate(self, context) -> np.ndarray:
        return self.inner_pattern(context)
