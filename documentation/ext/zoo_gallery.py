"""
Sphinx extension: pattern-method documentation zoo.

On ``builder-inited`` this extension:

1. Imports every pattern module and collects all :class:`PatternMethod`
   subclasses that carry a ``zoo_meta`` class variable.
2. For each such class it:
   a. Loads the declared source image (raw TIF + optional label TIF) from
      ``documentation/zoo_sources/``.
   b. Creates a *zoo subclass* that pre-binds ``get_um_meshgrid`` /
      ``center_um`` to the sample image geometry.
   c. Instantiates the class with the kwargs in ``zoo_meta``, builds a
      :class:`ZooContext`, and calls ``generate()``.
   d. Saves a side-by-side overlay PNG to
      ``documentation/_generated/zoo/<ClassName>.png``.
3. Writes ``documentation/_generated/zoo/gallery.md`` — a MyST markdown
   fragment that ``method_zoo.md`` includes.

Configuration (``conf.py``)
---------------------------
``zoo_sources_dir``  (default: ``<docs_dir>/zoo_sources``)
    Directory that contains the raw / label TIFs and ``metadata.toml``.
``zoo_output_dir``   (default: ``<docs_dir>/_generated/zoo``)
    Directory where PNGs and ``gallery.md`` are written.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sphinx.application import Sphinx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sphinx entry point
# ---------------------------------------------------------------------------


def setup(app: Sphinx) -> dict:
    app.add_config_value("zoo_sources_dir", None, "env")
    app.add_config_value("zoo_output_dir", None, "env")
    app.connect("builder-inited", _build_gallery)
    return {"version": "0.1", "parallel_read_safe": True}


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def _build_gallery(app: Sphinx) -> None:
    docs_dir = Path(app.srcdir)

    sources_dir = (
        Path(app.config.zoo_sources_dir)
        if app.config.zoo_sources_dir
        else docs_dir / "zoo_sources"
    )
    # PNGs land in _static/zoo/ so Sphinx copies them verbatim to the HTML
    # output _static/zoo/ directory.
    output_dir = (
        Path(app.config.zoo_output_dir)
        if app.config.zoo_output_dir
        else docs_dir / "_generated" / "zoo"
    )
    # The gallery fragment is written to the docs root so that image paths
    # inside it resolve correctly when included by method_zoo.md.
    fragment_dir = docs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sources_dir.exists():
        logger.warning(
            "zoo_gallery: sources directory %s not found; skipping zoo build.",
            sources_dir,
        )
        _write_placeholder(fragment_dir)
        return

    try:
        import matplotlib
        import tifffile
    except ImportError as exc:
        logger.warning("zoo_gallery: missing dependency (%s); skipping zoo build.", exc)
        _write_placeholder(fragment_dir)
        return

    metadata = _load_metadata(sources_dir)
    classes = _discover_zoo_classes()

    if not classes:
        logger.warning("zoo_gallery: no PatternMethod subclasses with zoo_meta found.")
        _write_placeholder(fragment_dir)
        return

    known_models = _load_known_models()

    entries: list[dict] = []
    for cls in classes:
        try:
            entry = _build_entry(cls, sources_dir, output_dir, metadata, known_models)
            if entry:
                entries.append(entry)
        except Exception as exc:
            logger.warning(
                "zoo_gallery: skipping %s - %s: %s",
                cls.__name__,
                type(exc).__name__,
                exc,
            )

    _write_gallery_md(fragment_dir, output_dir, entries)
    logger.info(
        "zoo_gallery: wrote %d gallery entries to %s.", len(entries), output_dir
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
_PATTERN_MODULES = [
    "pyclm.core.patterns.embryo_patterns",
    "pyclm.core.patterns.bar_patterns",
    "pyclm.core.patterns.cell_intensity_patterns",
    "pyclm.core.patterns.fbc_cell_movement",
    "pyclm.core.patterns.wave_patterns",
    "pyclm.core.patterns.static_patterns",
]


def _load_known_models() -> dict:
    """Return the inverted known_models map: {cls: toml_key}."""
    try:
        from pyclm.core.patterns import known_models

        return {cls: key for key, cls in known_models.items()}
    except ImportError:
        return {}


def _get_toml_name(cls: type, known_models_inv: dict) -> str | None:
    """Look up the TOML registration key for *cls*.

    Matches first by direct class identity, then by the ``name`` class
    attribute (handles cases like ``BarPattern`` → ``BarPatternBase`` → ``"bar"``).
    """
    if cls in known_models_inv:
        return known_models_inv[cls]
    name_attr = getattr(cls, "name", None)
    if name_attr:
        # name_attr is the TOML key when the class is registered directly
        for klass, key in known_models_inv.items():
            if getattr(klass, "name", None) == name_attr:
                return key
    return None


def _discover_zoo_classes() -> list[type]:
    """Import pattern modules in _PATTERN_MODULES order and return classes that
    have ``zoo_meta``, preserving that order."""
    # Make pyclm importable from the Sphinx source tree.
    src_dir = str(Path(__file__).parents[2] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    zoo_classes: list[type] = []
    seen: set[type] = set()

    for mod_name in _PATTERN_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            logger.warning("zoo_gallery: could not import %s - %s", mod_name, exc)
            continue

        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and "zoo_meta" in obj.__dict__
                and obj.__dict__["zoo_meta"] is not None
                and obj not in seen
            ):
                zoo_classes.append(obj)
                seen.add(obj)

    return zoo_classes


# ---------------------------------------------------------------------------
# Per-entry build
# ---------------------------------------------------------------------------


def _build_entry(
    cls: type,
    sources_dir: Path,
    output_dir: Path,
    metadata: dict,
    known_models_inv: dict,
) -> dict | None:
    from pyclm.core.patterns.zoo import ZooContext, make_zoo_subclass

    zoo_meta = cls.zoo_meta
    source = zoo_meta.source
    pixel_size_um = metadata.get(source, {}).get("pixel_size_um", 1.0)
    vmin, vmax = (
        metadata.get(source, {}).get("vmin"),
        metadata.get(source, {}).get("vmax"),
    )

    # load and normalize raw image and segmentation
    raw, seg = _load_source(sources_dir, source)
    raw = np.clip(raw, vmin, vmax) if (vmin is not None and vmax is not None) else raw
    raw = (raw - raw.min()) / (raw.max() - raw.min()) * 255

    # Build zoo subclass with geometry bound to the sample image
    ZooCls = make_zoo_subclass(cls, raw.shape, pixel_size_um)

    # use all kwargs
    kwargs = zoo_meta.kwargs
    real_kwargs = {}
    public_kwargs = {}
    for item, kwarg in kwargs.items():
        real_kwargs[item.lstrip("_")] = kwarg
        if item[0] != "_":
            public_kwargs[item] = kwarg

    # Instantiate
    instance = ZooCls(**real_kwargs)
    instance.pixel_size_um = pixel_size_um
    instance.pattern_shape = raw.shape

    # Build context
    context = ZooContext(
        raw_image=raw, seg_image=seg, time_seconds=zoo_meta.time_seconds
    )

    # Generate pattern
    pattern = np.asarray(instance.generate(context), dtype=float)
    pattern = (np.clip(pattern, 0, 1) * 255).astype(np.uint8)

    # Render overlay PNG
    png_name = f"{cls.__name__}.png"
    png_path = output_dir / png_name
    _render_overlay(raw, pattern, png_path)

    title = zoo_meta.title or getattr(cls, "name", cls.__name__)
    description = zoo_meta.description or (inspect.getdoc(cls) or "").split("\n")[0]

    return {
        "cls_name": cls.__name__,
        "module": cls.__module__,
        "png_name": png_name,
        "title": title,
        "description": description,
        "kwargs": public_kwargs,
        "toml_name": _get_toml_name(cls, known_models_inv),
    }


def _load_source(sources_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    import tifffile

    raw_path = sources_dir / f"{source}.tif"
    if not raw_path.exists():
        raise FileNotFoundError(f"Source image not found: {raw_path}")

    raw = tifffile.imread(str(raw_path))
    if raw.ndim == 3:
        raw = raw[0]
    raw = raw.astype(np.float32)

    seg_path = sources_dir / f"{source}_seg.tif"
    seg = tifffile.imread(str(seg_path))
    if seg.ndim == 3:
        seg = seg[0]
    seg = seg.astype(np.int32)

    return raw, seg


def _load_metadata(sources_dir: Path) -> dict:
    meta_path = sources_dir / "metadata.toml"
    import tomllib  # Python ≥ 3.11

    with open(meta_path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_overlay(raw: np.ndarray, pattern: np.ndarray, output_path: Path) -> None:
    import cv2

    alpha = 0.4

    base_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    cyan_pattern = np.zeros_like(base_bgr)
    cyan_pattern[:, :, 0] = pattern
    cyan_pattern[:, :, 1] = pattern
    cyan_pattern[:, :, 2] = 0

    # Blend the base image with the cyan pattern
    blended = cv2.addWeighted(base_bgr, 1 - alpha, cyan_pattern, alpha, 0)

    print(blended.shape)

    cv2.imwrite(str(output_path), blended)


# ---------------------------------------------------------------------------
# Gallery markdown generation
# ---------------------------------------------------------------------------


def _write_gallery_md(
    fragment_dir: Path, output_dir: Path, entries: list[dict]
) -> None:
    """Write _zoo_gallery.md to *fragment_dir* (docs root).

    Image paths are written as ``_generated/zoo/<name>.png`` - relative to
    *fragment_dir* - so they resolve correctly when Sphinx processes the
    gallery fragment as part of ``method_zoo.md``.
    """
    # Compute the relative path prefix from fragment_dir to output_dir.
    try:
        rel_prefix = output_dir.relative_to(fragment_dir).as_posix()
    except ValueError:
        rel_prefix = str(output_dir)

    lines = [
        "<!-- Auto-generated by zoo_gallery extension - do not edit manually -->",
        "",
    ]

    for entry in entries:
        img_path = f"{rel_prefix}/{entry['png_name']}"
        fqn = f"{entry['module']}.{entry['cls_name']}"
        api_ref = f"{{py:class}}`{entry['cls_name']} <{fqn}>`"

        lines += [
            f"### {entry['title']}",
            "",
            f"![{entry['title']}]({img_path})",
            "",
        ]
        if entry["description"]:
            lines += [entry["description"], ""]

        # TOML key + API cross-reference line
        meta_parts = []
        if entry["toml_name"]:
            meta_parts.append(f'**TOML key:** `"{entry["toml_name"]}"`')
        meta_parts.append(f"**API:** {api_ref}")
        lines += [" · ".join(meta_parts), ""]

        if entry["kwargs"]:
            lines += [
                "**Parameters used in this example:**",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
            ]
            for k, v in entry["kwargs"].items():
                lines.append(f"| `{k}` | `{v!r}` |")
            lines.append("")

        if entry is not entries[-1]:
            lines.append("---")
            lines.append("")
        else:
            lines.append("")

    (fragment_dir / "_zoo_gallery.md").write_text("\n".join(lines), encoding="utf-8")


def _write_placeholder(fragment_dir: Path) -> None:
    fragment_dir.mkdir(parents=True, exist_ok=True)
    (fragment_dir / "_zoo_gallery.md").write_text(
        "<!-- Zoo gallery could not be built (missing sources or dependencies). -->\n",
        encoding="utf-8",
    )
