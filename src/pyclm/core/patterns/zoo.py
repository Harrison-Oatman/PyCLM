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
            group="moving",
        )

The ``source`` field names a sample image stored in
``documentation/zoo_sources/<source>.tif`` with its label image
``<source>_seg.tif``.  ``kwargs`` are passed verbatim to the pattern
constructor; a key with a leading underscore is passed but left out of the
TOML shown on the page.

A registered method that is another method with fixed arguments does not
get a card of its own; it names that method instead and is listed on its
card::

    class GlobalCycleModel(PatternMethod):
        zoo_meta = ZooMeta(
            same_as="global_pulse",
            equivalent="period_min = 2 * period_m, on_min = period_m",
        )

The gallery is built from ``known_models``; ``tests/test_zoo_gallery.py``
fails when a registered method has no ``zoo_meta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# ZooMeta
# ---------------------------------------------------------------------------

# The gallery's families, in page order: (key, heading, one-line blurb).
ZOO_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("static", "Static shapes", "Fixed patterns that never change."),
    (
        "moving",
        "Moving stripes",
        "Open-loop patterns that travel across the field.",
    ),
    (
        "timed",
        "Timed and blinking",
        "Open-loop stimulation that switches on and off on a schedule.",
    ),
    (
        "per_cell",
        "Per-cell direction",
        "Closed loop: light one side of every segmented cell.",
    ),
    (
        "per_nucleus",
        "Per-nucleus feedback",
        "Closed loop: light each nucleus according to its own intensity.",
    ),
    (
        "embryo",
        "Embryo axis",
        "Closed loop: light a fraction of the largest object's long axis.",
    ),
    ("composite", "Combining methods", "One method per half of the field."),
    (
        "adaptive",
        "Adaptive acquisition",
        "Wrap any method and change the run: move the stage, adjust an "
        "exposure or a device property.",
    ),
    (
        "diagnostic",
        "Diagnostics",
        "Checks of the setup itself (DMD calibration on a fluorescent slide), "
        "not experiments on cells.",
    ),
)


@dataclass
class ZooMeta:
    """Declares how a PatternMethod should appear in the documentation zoo.

    Parameters
    ----------
    source:
        Name of the sample image set, e.g. ``"mdck"``, ``"fly"``, or
        ``"mcf10a"``.  ``documentation/zoo_sources/<source>.tif`` and
        ``<source>_seg.tif`` must exist.  The method sees the image in
        camera counts.
    kwargs:
        Keyword arguments forwarded to the pattern-method constructor.
    time_seconds:
        Experiment time (seconds) of the thumbnail and the main panels.
    title:
        Gallery-card heading.  Defaults to the class ``name`` attribute.
    description:
        One-sentence description shown beneath the image.
    group:
        Key of the family the card is listed under (see ``ZOO_GROUPS``).
    times:
        Experiment times (seconds) for a filmstrip of the pattern, for a
        method that moves.
    timeline:
        ``(start, stop)`` in seconds: plot the lit fraction of the field
        over that stretch, for a method that switches on and off.
    same_as:
        Registered name of the method this one duplicates; no card of its
        own, it is listed on that method's card.
    equivalent:
        The arguments of ``same_as`` that give this method's pattern.
    pending:
        Set while the card waits for sample data the zoo does not have yet
        (a time series, say): what is needed. The gallery lists the method
        with this note instead of a card.
    """

    source: str = ""
    kwargs: dict = field(default_factory=dict)
    time_seconds: float = 0.0
    title: str = ""
    description: str = ""
    group: str = ""
    times: tuple[float, ...] = ()
    timeline: tuple[float, float] | None = None
    same_as: str = ""
    equivalent: str = ""
    pending: str = ""


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

    def grid(self):
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
