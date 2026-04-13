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

    def segmentation(self, channel_name: str) -> np.ndarray:
        return self._seg

    def stim_raw(self) -> np.ndarray:
        return self._raw

    def stim_seg(self) -> np.ndarray:
        return self._seg


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
