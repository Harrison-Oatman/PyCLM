"""
The vocabulary of frame-derived data kinds.

A kind names what a piece of data *is*: ``"raw"`` (a camera frame),
``"seg"`` (a label image from the experiment's default ``[segmentation]``
table), ``"seg:<name>"`` (a label image from a named
``[segmentation.<name>]`` table) or ``"tracks"`` (a label image relabelled
with track ids plus a table). The router keys its table on
``(experiment, channel, kind)``; producers are registered by the *base*
kind (``"seg"`` for every segmentation), so one segmentation process serves
every named segmentation.
"""

from __future__ import annotations

DEFAULT_SEGMENTATION = "segmentation"
BASE_KINDS = ("raw", "seg", "tracks")


def base_kind(kind: str) -> str:
    """``"seg"`` for ``"seg:nuclei"``; other kinds unchanged."""
    return kind.split(":", 1)[0]


def seg_kind(name: str | None = None) -> str:
    """Routing kind of a segmentation: ``"seg"`` for the default, ``"seg:<name>"`` otherwise."""
    if not name or name == DEFAULT_SEGMENTATION:
        return "seg"
    return f"seg:{name}"


def seg_name(kind: str) -> str:
    """Segmentation name of a ``seg`` kind (``"segmentation"`` for the default)."""
    if kind == "seg":
        return DEFAULT_SEGMENTATION
    if kind.startswith("seg:") and len(kind) > 4:
        return kind[4:]
    raise ValueError(f"{kind!r} is not a segmentation kind")


def is_seg_kind(kind: str) -> bool:
    return base_kind(kind) == "seg"
