"""
Zoo metadata and context for the pattern-method documentation gallery.

Usage
-----
Attach a ``zoo_meta`` class variable to any :class:`PatternMethod` subclass
to include it in the generated gallery::

    class MyPattern(PatternMethod):
        zoo_meta = ZooMeta(
            source="mdck",
            kwargs={"period": 40, "duty_cycle": 0.3},
            time_seconds=120.0,
            title="My Pattern",
            description="A one-line description shown in the gallery card.",
        )

The ``source`` field names a sample image stored in
``documentation/zoo_sources/<source>.tif`` (and optionally
``<source>_seg.tif`` for a label image).  ``kwargs`` are passed verbatim
to the pattern constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# ZooMeta
# ---------------------------------------------------------------------------


@dataclass
class ZooMeta:
    """Declares how a PatternMethod should appear in the documentation zoo.

    Parameters
    ----------
    source:
        Name of the sample image set, e.g. ``"mdck"``, ``"fly"``, or
        ``"mcf10a"``.  A file ``documentation/zoo_sources/<source>.tif``
        must exist; ``<source>_seg.tif`` is optional (synthetic blobs are
        used when absent).
    kwargs:
        Keyword arguments forwarded to the pattern-method constructor.
    time_seconds:
        Experiment time (seconds) injected into the context.
    title:
        Gallery-card heading.  Defaults to the class ``name`` attribute.
    description:
        One-sentence description shown beneath the image.
    """

    source: str
    kwargs: dict = field(default_factory=dict)
    time_seconds: float = 0.0
    title: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# ZooContext  -  minimal PatternContext substitute
# ---------------------------------------------------------------------------


class ZooContext:
    """Minimal substitute for :class:`PatternContext` used during zoo builds.

    Any channel name is accepted; all channels map to the same raw/seg
    images supplied at construction time.

    Parameters
    ----------
    raw_image:
        2-D uint16 (or similar) array representing a raw fluorescence image.
    seg_image:
        2-D integer label array (0 = background, >0 = cell id).
    time_seconds:
        Value exposed as ``context.time``.
    """

    def __init__(
        self,
        raw_image: np.ndarray,
        seg_image: np.ndarray,
        time_seconds: float = 0.0,
    ) -> None:
        self._raw = raw_image
        self._seg = seg_image
        self.time = time_seconds

    # --- PatternContext interface (channel-agnostic variants) ---------------

    def raw(self, channel_name: str) -> np.ndarray:
        return self._raw

    def segmentation(self, channel_name: str, name: str = "segmentation") -> np.ndarray:
        return self._seg

    def regions(self, channel_name: str, name: str = "segmentation"):
        from ..measure import Regions

        return Regions(self._seg)

    # --- runtime settings: recorded, never applied, in the zoo ---------------
    @property
    def requests(self) -> list:
        return list(getattr(self, "_requests", []))

    def settings(self, channel_name: str) -> dict:
        return {
            "exposure_ms": 10.0,
            "binning": 1,
            "config_groups": {},
            "device_properties": {},
        }

    def position(self):
        return None

    def _remember(self, change):
        self.__dict__.setdefault("_requests", []).append(change)

    def set_exposure(self, channel_name, ms):
        self._remember(("exposure", channel_name, "exposure_ms", ms))

    def set_config(self, channel_name, group, preset):
        self._remember(("config", channel_name, group, preset))

    def set_property(self, channel_name, device, prop, value):
        self._remember(("property", channel_name, f"{device}-{prop}", value))

    def set_position(self, x=None, y=None, z=None, pfs_offset=None):
        for key, value in (("x", x), ("y", y), ("z", z), ("pfs_offset", pfs_offset)):
            if value is not None:
                self._remember(("position", None, key, value))

    def stim_raw(self) -> np.ndarray:
        return self._raw

    def stim_seg(self) -> np.ndarray:
        return self._seg

    # --- Stage 3 additions: timepoint, history, tracks, previous patterns ---

    t = 0
    generation = 0

    def tracks(self, channel_name: str):
        from skimage.measure import regionprops_table

        from ..tracking import TrackRow, Tracks

        props = regionprops_table(
            np.asarray(self._seg).astype(np.int64),
            properties=("label", "centroid", "area"),
        )
        rows = [
            TrackRow(int(lab), int(lab), float(y), float(x), int(a), 0)
            for lab, y, x, a in zip(
                props["label"],
                props["centroid-0"],
                props["centroid-1"],
                props["area"],
                strict=True,
            )
        ]
        return Tracks(self._seg, rows)

    def history(self, channel_name: str, kind: str = "seg", n=None) -> list:
        if kind == "tracks":
            return [self.tracks(channel_name)]
        return [self._seg if kind.startswith("seg") else self._raw]

    def stim_history(self, kind: str = "raw", n=None) -> list:
        return self.history("", kind, n)

    def last_pattern(self):
        return None

    def pattern_history(self, n=None) -> list:
        return []


# ---------------------------------------------------------------------------
# Zoo subclass factory
# ---------------------------------------------------------------------------


def make_zoo_subclass(cls: type, pattern_shape: tuple[int, int], pixel_size_um: float):
    """Return a zoo-safe subclass of *cls*.

    The subclass pre-computes the um-coordinate meshgrid from the sample
    image geometry and overrides :meth:`get_um_meshgrid` and
    :meth:`center_um` so that ``generate()`` is fully independent of any
    hardware state that would normally be injected by
    :meth:`PatternMethod.configure_system`.

    Parameters
    ----------
    cls:
        A :class:`PatternMethod` subclass.
    pattern_shape:
        ``(height, width)`` of the sample image in pixels.
    pixel_size_um:
        Physical pixel size in micrometres (from the source metadata).

    Returns
    -------
    type
        A new class (not an instance) ready to be instantiated with the
        pattern's ``zoo_meta.kwargs``.
    """
    h, w = pattern_shape
    y_range = np.arange(h) * pixel_size_um
    x_range = np.arange(w) * pixel_size_um
    _xx, _yy = np.meshgrid(x_range, y_range)
    _center = (h * pixel_size_um / 2.0, w * pixel_size_um / 2.0)

    class _ZooVariant(cls):
        def get_um_meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
            return _xx, _yy

        def center_um(self) -> tuple[float, float]:
            return _center

    _ZooVariant.__name__ = f"{cls.__name__}ZooVariant"
    _ZooVariant.__qualname__ = _ZooVariant.__name__
    return _ZooVariant
