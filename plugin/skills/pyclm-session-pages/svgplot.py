"""
Line charts as inline SVG in the session-page style, with nothing to install.

Plain Python (numpy arrays work too; anything iterable of numbers). The
result is a string to paste into a page inside a ``.figure`` block. It
scales with the page, stays sharp when zoomed, and its text is text, so the
page's fonts apply.

    import sys; sys.path.insert(0, r"<plugin>/skills/pyclm-session-pages")
    from svgplot import line_chart

    svg = line_chart(
        [
            {"label": "bar60.01", "x": t_min, "y": ratio_1},
            {"label": "unlit.01", "x": t_min, "y": ratio_u, "color": "dark"},
        ],
        x_label="time (min)",
        y_label="545 in bar / outside",
        light=[(0, 30)],
    )

Each series: ``label``, ``x``, ``y``; optional ``color`` (a hex color,
``"dark"`` for a no-light control, or left out to take the next data
color), ``dashed``, ``points`` (draw markers), ``band`` (``(low, high)``
sequences drawn as a shaded range). Colors follow the condition: pass the
same color for the same condition on every page.

Copy the file next to the page's build script if the plugin folder is not
on the path; it has no imports outside the standard library.
"""

from __future__ import annotations

import math
from html import escape

DATA = (
    "#1e66f5",
    "#e0661a",
    "#0a9199",
    "#b98300",
    "#d85fb0",
    "#d20f39",
    "#8839ef",
    "#40a02b",
)
DARK = "#a59d8e"
INK, INK_2, MUTED = "#33302a", "#5c574d", "#787164"
GRID, LINE, SURFACE = "#e3dccd", "#d9d0bf", "#fbf9f4"
LIGHT_BAND, LIGHT_EDGE = "rgba(70,175,215,0.27)", "#38a9d1"
SANS = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
MONO = "IBM Plex Mono, Consolas, ui-monospace, monospace"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """About ``n`` round tick values covering ``lo`` to ``hi``."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0]
    if hi == lo:
        return [lo]
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda st: abs(st - raw))
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 12))
        v += step
    return ticks


def _fmt(v: float, step: float) -> str:
    if step >= 1 and abs(v - round(v)) < 1e-9:
        return f"{round(v):d}"
    exp = math.floor(math.log10(step))
    digits = max(0, -exp)
    if abs(step / 10**exp - 2.5) < 1e-9:
        digits += 1
    return f"{v:.{digits}f}"


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _range(values: list[float], pad: float) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    return lo - pad * span, hi + pad * span


_ids = iter(range(1, 1_000_000))


def line_chart(
    series: list[dict],
    *,
    x_label: str = "",
    y_label: str = "",
    light: list[tuple[float, float]] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    hline: float | None = None,
    width: int = 720,
    height: int = 300,
    title: str = "",
) -> str:
    """An SVG line chart (string). ``light`` shades x intervals light blue;
    ``hline`` draws a dashed reference line (for example 1 for a ratio)."""
    left, right, top, bottom = 58, 16, 30 if (title or len(series) > 1) else 14, 44
    pw, ph = width - left - right, height - top - bottom

    xs = _finite(v for s in series for v in s["x"])
    ys = _finite(v for s in series for v in s["y"])
    for s in series:
        if s.get("band"):
            ys += _finite(s["band"][0]) + _finite(s["band"][1])
    if hline is not None:
        ys.append(hline)
    if not xs or not ys:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    x0, x1 = x_range or _range(xs, 0.0)
    y0, y1 = y_range or _range(ys, 0.06)

    def X(v):
        return left + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return top + (1 - (v - y0) / (y1 - y0)) * ph

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:{width}px;display:block" '
        f'font-family="{SANS}" font-size="12">'
    ]
    if title:
        out.append(f"<title>{escape(title)}</title>")
    out.append(
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="{SURFACE}"/>'
    )

    for a, b in light or []:
        a, b = max(a, x0), min(b, x1)
        if b > a:
            out.append(
                f'<rect x="{X(a):.1f}" y="{top}" width="{X(b) - X(a):.1f}" height="{ph}" '
                f'fill="{LIGHT_BAND}" stroke="{LIGHT_EDGE}" stroke-width="1"/>'
            )

    xt, yt = nice_ticks(x0, x1), nice_ticks(y0, y1, 6)
    xs_step = xt[1] - xt[0] if len(xt) > 1 else 1.0
    ys_step = yt[1] - yt[0] if len(yt) > 1 else 1.0
    for v in yt:
        if y0 <= v <= y1:
            out.append(
                f'<line x1="{left}" x2="{left + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/>'
            )
            out.append(
                f'<text x="{left - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{MUTED}" '
                f'font-family="{MONO}" font-size="11">{_fmt(v, ys_step)}</text>'
            )
    for v in xt:
        if x0 <= v <= x1:
            out.append(
                f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top}" y2="{top + ph}" stroke="{GRID}"/>'
            )
            out.append(
                f'<text x="{X(v):.1f}" y="{top + ph + 16}" text-anchor="middle" fill="{MUTED}" '
                f'font-family="{MONO}" font-size="11">{_fmt(v, xs_step)}</text>'
            )
    out.append(
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="{LINE}"/>'
    )
    if hline is not None and y0 <= hline <= y1:
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{Y(hline):.1f}" y2="{Y(hline):.1f}" '
            f'stroke="{INK_2}" stroke-dasharray="4 3" stroke-width="1"/>'
        )

    clip = f"svgplot{next(_ids)}"
    out.append(
        f'<clipPath id="{clip}"><rect x="{left}" y="{top}" width="{pw}" height="{ph}"/></clipPath>'
    )
    legend = []
    k = 0
    for s in series:
        c = s.get("color")
        if c == "dark":
            c = DARK
        elif not c:
            c = DATA[k % len(DATA)]
            k += 1
        pts = [
            (float(a), float(b))
            for a, b in zip(s["x"], s["y"], strict=False)
            if a is not None
            and b is not None
            and math.isfinite(float(a))
            and math.isfinite(float(b))
        ]
        band = s.get("band")
        if band:
            lo = [(float(a), float(b)) for a, b in zip(s["x"], band[0], strict=False)]
            hi = [(float(a), float(b)) for a, b in zip(s["x"], band[1], strict=False)]
            poly = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in lo + hi[::-1])
            out.append(
                f'<polygon points="{poly}" fill="{c}" fill-opacity="0.15" clip-path="url(#{clip})"/>'
            )
        if len(pts) > 1:
            d = " ".join(
                f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}"
                for i, (a, b) in enumerate(pts)
            )
            dash = ' stroke-dasharray="5 3"' if s.get("dashed") else ""
            out.append(
                f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.8" '
                f'stroke-linejoin="round"{dash} clip-path="url(#{clip})"/>'
            )
        if s.get("points") or len(pts) == 1:
            for a, b in pts:
                out.append(
                    f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="2.6" fill="{c}"/>'
                )
        legend.append((s.get("label", ""), c, s.get("dashed")))

    if title:
        out.append(
            f'<text x="{left}" y="18" fill="{INK}" font-weight="600" font-size="13">{escape(title)}</text>'
        )
    if len(series) > 1:
        lx = left + pw
        for label, c, dashed in reversed(legend):
            w = 7 * len(label) + 26
            lx -= w
            dash = ' stroke-dasharray="5 3"' if dashed else ""
            out.append(
                f'<line x1="{lx:.0f}" x2="{lx + 16:.0f}" y1="14" y2="14" stroke="{c}" stroke-width="2"{dash}/>'
            )
            out.append(
                f'<text x="{lx + 20:.0f}" y="18" fill="{INK_2}">{escape(label)}</text>'
            )

    if x_label:
        out.append(
            f'<text x="{left + pw / 2:.0f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}">{escape(x_label)}</text>'
        )
    if y_label:
        out.append(
            f'<text transform="translate(14 {top + ph / 2:.0f}) rotate(-90)" text-anchor="middle" '
            f'fill="{INK_2}">{escape(y_label)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)
