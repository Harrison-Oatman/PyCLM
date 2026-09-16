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
"""

from __future__ import annotations

import logging

import numpy as np

from ..experiments import Experiment
from .pattern import AcquiredImageRequest, CameraProperties, PatternMethod

logger = logging.getLogger(__name__)

HORIZONTAL = ("left", "right")  # the boundary is vertical: halves side by side
VERTICAL = ("top", "bottom")  # the boundary is horizontal: halves one above the other
REGIONS = HORIZONTAL + VERTICAL


class SplitPattern(PatternMethod):
    """One pattern method per half of the field; see the module docstring."""

    name = "split"

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
        merged: dict = {}
        for child in self.children.values():
            for req in child.initialize(experiment):
                have = merged.get(req.id)
                if have is None:
                    merged[req.id] = req
                    continue
                merged[req.id] = AcquiredImageRequest(
                    req.id,
                    have.needs_raw or req.needs_raw,
                    have.needs_seg or req.needs_seg,
                    have.needs_tracks or req.needs_tracks,
                    max(have.history, req.history),
                    tuple(dict.fromkeys((*have.segmentations, *req.segmentations))),
                )
        return list(merged.values())

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
            pattern = np.asarray(child.generate(context), dtype=np.float32)
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
