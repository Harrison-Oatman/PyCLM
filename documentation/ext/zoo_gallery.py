"""
Sphinx extension: pattern-method documentation zoo.

On ``builder-inited`` this extension:

1. Walks ``known_models`` and finds the ``zoo_meta`` of every registered
   pattern method (on the class, or on the subclass a dispatching base such
   as ``bar`` builds). A method whose ``zoo_meta`` names ``same_as`` gets no
   card; it is listed on that method's card.
2. For each card it:
   a. Loads the declared source image (raw TIF + label TIF) from
      ``documentation/zoo_sources/``.
   b. Instantiates the method with the kwargs in ``zoo_meta`` (a string
      argument that names a file in ``zoo_sources/`` becomes its path), sets
      the sample geometry on it and, for a composite, on its parts.
   c. Calls ``generate()`` with a :class:`ZooContext` that holds the image in
      camera counts, at ``time_seconds`` and at each of ``times``.
   d. Saves a thumbnail and a detail figure (raw | pattern | overlay, plus a
      filmstrip or a timeline) to ``_generated/static/zoo/``, which is added
      to ``html_static_path`` so the pages find them under ``_static/zoo/``.
3. Writes ``_zoo_gallery.md``, a MyST fragment that ``method_zoo.md``
   includes: a thumbnail grid per family, then one entry per card.

:func:`build_gallery` does all of it without Sphinx; the tests call it.

Configuration (``conf.py``)
---------------------------
``zoo_sources_dir``  (default: ``<docs_dir>/zoo_sources``)
    Directory that contains the raw / label TIFs and ``metadata.toml``.
``zoo_output_dir``   (default: ``<docs_dir>/_generated/static/zoo``)
    Directory where the PNGs are written; its parent is added to
    ``html_static_path``.
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sphinx.application import Sphinx

logger = logging.getLogger(__name__)

# the overlay: the pattern in cyan over the grey image
_CYAN = np.array([0.0, 0.85, 0.85])
_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Sphinx entry point
# ---------------------------------------------------------------------------


def setup(app: Sphinx) -> dict:
    app.add_config_value("zoo_sources_dir", None, "env")
    app.add_config_value("zoo_output_dir", None, "env")
    app.connect("builder-inited", _on_builder_inited)
    return {"version": "0.2", "parallel_read_safe": True}


def _on_builder_inited(app: Sphinx) -> None:
    docs_dir = Path(app.srcdir)
    sources_dir = Path(app.config.zoo_sources_dir or docs_dir / "zoo_sources")
    output_dir = Path(
        app.config.zoo_output_dir or docs_dir / "_generated" / "static" / "zoo"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # served as _static/<output_dir.name>/
    static_root = str(output_dir.parent)
    if static_root not in app.config.html_static_path:
        app.config.html_static_path.append(static_root)

    fragment = docs_dir / "_zoo_gallery.md"
    try:
        entries = build_gallery(sources_dir, output_dir, fragment, docs_dir)
    except Exception as exc:
        logger.warning(
            "zoo_gallery: skipping zoo build - %s: %s", type(exc).__name__, exc
        )
        fragment.write_text(
            "<!-- Zoo gallery could not be built (missing sources or dependencies). -->\n",
            encoding="utf-8",
        )
        return
    logger.info(
        "zoo_gallery: wrote %d gallery entries to %s.", len(entries), output_dir
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_gallery(
    sources_dir: Path, output_dir: Path, fragment: Path, docs_dir: Path
) -> list[dict]:
    """Render every card into *output_dir* and write the MyST *fragment*.

    Image paths in the fragment are relative to *docs_dir* (the page that
    includes it lives there). Returns the card entries.
    """
    import tomllib

    import matplotlib

    matplotlib.use("Agg")

    src_dir = str(Path(__file__).parents[2] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from pyclm.core.patterns import known_models

    with open(sources_dir / "metadata.toml", "rb") as f:
        metadata = tomllib.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    cards, aliases = collect(known_models)

    entries = []
    for key, meta in cards:
        try:
            entry = _build_entry(
                key, meta, known_models, sources_dir, output_dir, metadata
            )
        except Exception as exc:
            logger.warning(
                "zoo_gallery: skipping %s - %s: %s", key, type(exc).__name__, exc
            )
            continue
        entry["aliases"] = aliases.get(key, [])
        entries.append(entry)

    _write_fragment(fragment, output_dir, docs_dir, entries)
    return entries


def zoo_meta_for(key: str, cls: type):
    """The ``zoo_meta`` declared for registered name *key*, or None.

    Looks on *cls* itself, then on its subclasses whose ``name`` is *key*
    (``bar`` registers ``BarPatternBase``, which builds a ``BarPattern``).
    """
    if cls.__dict__.get("zoo_meta") is not None:
        return cls.__dict__["zoo_meta"]
    pending = list(cls.__subclasses__())
    while pending:
        sub = pending.pop()
        if (
            getattr(sub, "name", None) == key
            and sub.__dict__.get("zoo_meta") is not None
        ):
            return sub.__dict__["zoo_meta"]
        pending.extend(sub.__subclasses__())
    return None


def collect(registry: dict[str, type]):
    """Split the registered names into cards and aliases.

    Returns ``([(key, zoo_meta), ...] in family order, {card key: [(alias,
    equivalent), ...]})``. Names without ``zoo_meta`` are left out (the tests
    fail on them).
    """
    from pyclm.core.patterns.zoo import ZOO_GROUPS

    order = {g[0]: i for i, g in enumerate(ZOO_GROUPS)}
    cards, aliases = [], {}
    for key, cls in registry.items():
        meta = zoo_meta_for(key, cls)
        if meta is None:
            continue
        if meta.same_as:
            aliases.setdefault(meta.same_as, []).append((key, meta.equivalent))
        else:
            cards.append((key, meta))
    cards.sort(key=lambda c: order.get(c[1].group, len(order)))
    return cards, aliases


def _build_entry(key, meta, registry, sources_dir, output_dir, metadata) -> dict:
    from pyclm.core.patterns.zoo import ZooContext

    info = metadata.get(meta.source, {})
    pixel_size_um = info.get("pixel_size_um", 1.0)
    raw, seg = _load_source(sources_dir, meta.source)
    display = _display_image(raw, info.get("vmin"), info.get("vmax"))

    kwargs = {k.lstrip("_"): _resolve(v, sources_dir) for k, v in meta.kwargs.items()}
    method = registry[key](**kwargs)
    method.bind_registry(registry)
    _set_geometry(method, raw.shape, pixel_size_um)

    def pattern_at(t):
        context = ZooContext(raw_image=raw, seg_image=seg, time_seconds=t)
        out = np.asarray(method.generate(context), dtype=float)
        return np.clip(out, 0, 1)

    pattern = pattern_at(meta.time_seconds)
    thumb = f"{key}.png"
    _save_rgb(output_dir / thumb, _overlay(display, pattern))

    strip = [(t, _overlay(display, pattern_at(t))) for t in meta.times]
    timeline = None
    if meta.timeline:
        ts = np.linspace(meta.timeline[0], meta.timeline[1], 400)
        timeline = (ts, [float(pattern_at(t).mean()) for t in ts])

    detail = f"{key}_detail.png"
    _render_detail(
        output_dir / detail, display, pattern, meta.time_seconds, strip, timeline
    )

    cls = type(method)
    return {
        "key": key,
        "title": meta.title or key,
        "group": meta.group,
        "description": meta.description,
        "thumb": thumb,
        "detail": detail,
        "kwargs": {k: v for k, v in meta.kwargs.items() if not k.startswith("_")},
        "needs": _needs(method),
        "varies": bool(meta.times or meta.timeline),
        "api": f"{cls.__module__}.{cls.__name__}",
        "cls_name": cls.__name__,
    }


def _resolve(value, sources_dir: Path):
    """A string naming a file in the sources folder becomes its path."""
    if isinstance(value, str) and value and (sources_dir / value).is_file():
        return str(sources_dir / value)
    return value


def _set_geometry(method, shape, pixel_size_um) -> None:
    method.pixel_size_um = pixel_size_um
    method.pattern_shape = shape
    for child in getattr(method, "children", {}).values():
        _set_geometry(child, shape, pixel_size_um)


def _needs(method) -> list[str]:
    """What the method asks the pipeline for, in words."""
    needs = []
    for _channel, raw, seg_names, tracks, _history in method._requirements_list:
        if seg_names and "segmentation" not in needs:
            needs.append("segmentation")
        if tracks and "tracks" not in needs:
            needs.append("tracks")
        if raw and "raw image" not in needs:
            needs.append("raw image")
    if method._stim_requested:
        needs.append("stimulation image")
    return needs


def _load_source(sources_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    import tifffile

    raw = tifffile.imread(str(sources_dir / f"{source}.tif"))
    seg = tifffile.imread(str(sources_dir / f"{source}_seg.tif"))
    if raw.ndim == 3:
        raw = raw[0]
    if seg.ndim == 3:
        seg = seg[0]
    return raw.astype(np.float32), seg.astype(np.int32)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _display_image(raw, vmin=None, vmax=None) -> np.ndarray:
    """The raw image scaled to 0-1 for display (the method sees counts)."""
    lo = raw.min() if vmin is None else vmin
    hi = raw.max() if vmax is None else vmax
    return np.clip((raw - lo) / max(hi - lo, 1e-9), 0, 1)


def _overlay(display: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    grey = np.repeat(display[..., np.newaxis], 3, axis=-1)
    weight = _ALPHA * pattern[..., np.newaxis]
    return np.clip(grey * (1 - weight) + _CYAN * weight, 0, 1)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.imsave(path, rgb)


def _minutes(t_seconds: float) -> str:
    minutes = t_seconds / 60
    if minutes >= 120 and minutes % 60 == 0:
        return f"t = {minutes / 60:g} h"
    return f"t = {minutes:g} min"


def _frame(ax) -> None:
    """No ticks, a thin grey frame (so an all-white pattern still shows)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b0b0b0")
        spine.set_linewidth(0.8)


def _render_detail(path, display, pattern, t_seconds, strip, timeline) -> None:
    """raw | pattern | overlay, then the filmstrip or the timeline below."""
    import matplotlib.pyplot as plt

    extra = 1 if (strip or timeline) else 0
    ncols = max(3, len(strip))
    fig = plt.figure(figsize=(2.4 * ncols, 2.6 * (1 + extra)), dpi=110)
    grid = fig.add_gridspec(1 + extra, ncols * 2)
    width = (ncols * 2) // 3

    panels = [
        (display, "gray", "sample image"),
        (pattern, "gray", "pattern (DMD)"),
        (_overlay(display, pattern), None, "overlay"),
    ]
    for i, (img, cmap, label) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i * width : (i + 1) * width])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=10)
        _frame(ax)

    if strip:
        for i, (t, rgb) in enumerate(strip):
            ax = fig.add_subplot(grid[1, i * 2 : i * 2 + 2])
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(_minutes(t), fontsize=10)
            _frame(ax)
    elif timeline:
        ts, lit = timeline
        ax = fig.add_subplot(grid[1, :])
        ax.fill_between(ts / 60, lit, step="post", color=_CYAN, alpha=0.6, lw=0)
        ax.plot(ts / 60, lit, drawstyle="steps-post", color="#007a7a", lw=1)
        ax.set_xlabel("experiment time (min)", fontsize=9)
        ax.set_ylabel("field lit", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(ts[0] / 60, ts[-1] / 60)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gallery markdown generation
# ---------------------------------------------------------------------------


def _anchor(key: str) -> str:
    return "zoo-" + key.replace("_", "-")


def toml_block(key: str, kwargs: dict) -> str:
    """The ``[pattern]`` table that reproduces the example."""
    lines = ["[pattern]", f'method = "{key}"']
    tables = []
    for name, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, dict):
            tables.append((name, value))
        else:
            lines.append(f"{name} = {_toml_value(value)}")
    for name, table in tables:
        lines += ["", f"[pattern.{name}]"]
        lines += [f"{k} = {_toml_value(v)}" for k, v in table.items() if v is not None]
    return "\n".join(lines)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return repr(value)


def _write_fragment(fragment: Path, output_dir: Path, docs_dir: Path, entries) -> None:
    from pyclm.core.patterns.zoo import ZOO_GROUPS

    # thumbnails are raw HTML, so they use the served path (_static/<name>/);
    # detail figures are Markdown images, which Sphinx copies itself
    static_prefix = f"_static/{output_dir.name}"
    try:
        rel_prefix = output_dir.relative_to(docs_dir).as_posix()
    except ValueError:
        rel_prefix = output_dir.as_posix()

    groups = [
        (g, title, blurb, [e for e in entries if e["group"] == g])
        for g, title, blurb in ZOO_GROUPS
    ]
    known = {g[0] for g in ZOO_GROUPS}
    other = [e for e in entries if e["group"] not in known]
    if other:
        groups.append(("other", "Other", "", other))
    groups = [g for g in groups if g[3]]

    lines = [
        "<!-- Auto-generated by the zoo_gallery extension - do not edit manually -->",
        "",
        "## At a glance",
        "",
        "Grouped by family; click a card for the example's parameters and its pattern over time.",
        "",
    ]
    # one grid, in family order, each card labelled with its family
    lines.append('<div class="zoo-grid">')
    for _g, title, _blurb, members in groups:
        for e in members:
            lines.append(_card_html(e, title, static_prefix))
    lines += ["</div>", ""]

    lines += ["## Methods", ""]
    for _g, title, blurb, members in groups:
        lines += [f"### {title}", ""]
        if blurb:
            lines += [blurb, ""]
        for e in members:
            lines += _entry_md(e, rel_prefix)

    fragment.write_text("\n".join(lines), encoding="utf-8")


def _card_html(e: dict, family: str, static_prefix: str) -> str:
    title = html.escape(e["title"])
    tags = [_needs_label(e["needs"])]
    if e["varies"]:
        tags.append("changes over time")
    tag_html = "".join(f'<span class="zoo-tag">{html.escape(t)}</span>' for t in tags)
    also = ""
    if e["aliases"]:
        names = ", ".join(a for a, _ in e["aliases"])
        also = f'<span class="zoo-also">also {html.escape(names)}</span>'
    return (
        f'<a class="zoo-card" href="#{_anchor(e["key"])}">'
        f'<span class="zoo-family">{html.escape(family)}</span>'
        f'<img src="{static_prefix}/{e["thumb"]}" alt="{title}" loading="lazy">'
        f'<span class="zoo-title">{title}</span>'
        f'<code class="zoo-key">{html.escape(e["key"])}</code>'
        f'<span class="zoo-tags">{tag_html}</span>{also}</a>'
    )


def _needs_label(needs: list[str]) -> str:
    if not needs:
        return "open loop"
    return " + ".join(needs)


def _entry_md(e: dict, rel_prefix: str) -> list[str]:
    api = f"{{py:class}}`{e['cls_name']} <{e['api']}>`"
    lines = [
        f"({_anchor(e['key'])})=",
        f"#### {e['title']}",
        "",
        f"![{e['title']}]({rel_prefix}/{e['detail']})",
        "",
    ]
    if e["description"]:
        lines += [e["description"], ""]
    lines += [
        f'**TOML key:** `"{e["key"]}"` · **Needs:** {_needs_label(e["needs"])} · **API:** {api}',
        "",
        "```toml",
        toml_block(e["key"], e["kwargs"]),
        "```",
        "",
    ]
    for alias, equivalent in e["aliases"]:
        lines += [
            f'**Also registered:** `"{alias}"` is this method with `{equivalent}`.',
            "",
        ]
    return lines
