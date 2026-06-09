"""Hand-written SVG of the Hill q1 trend (no matplotlib, no external deps)."""

from __future__ import annotations

from pathlib import Path

from ..judge import Diagnosis

_W, _H = 640, 360
_PAD = 50.0


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def diagnosis_to_svg(diag: Diagnosis) -> str:
    """Render the primary resolution's Hill q1 trend as an SVG string."""
    res = diag.resolutions[0] if diag.resolutions else None
    title = f"evorare: {diag.verdict}"
    if res is None or len(res.points) == 0:
        body = f'<text x="{_PAD}" y="{_H / 2}" font-family="sans-serif">no data</text>'
        return _frame(title, body)

    xs = [float(p.generation) for p in res.points]
    ys = [float(p.hill_q1) for p in res.points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0

    def sx(x: float) -> float:
        return _PAD + (x - xmin) / xr * (_W - 2 * _PAD)

    def sy(y: float) -> float:
        return _H - _PAD - (y - ymin) / yr * (_H - 2 * _PAD)

    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys, strict=True))
    dots = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="#1f77b4"/>'
        for x, y in zip(xs, ys, strict=True)
    )
    axes = (
        f'<line x1="{_PAD}" y1="{_H - _PAD}" x2="{_W - _PAD}" y2="{_H - _PAD}" '
        f'stroke="#333"/>'
        f'<line x1="{_PAD}" y1="{_PAD}" x2="{_PAD}" y2="{_H - _PAD}" stroke="#333"/>'
    )
    labels = (
        f'<text x="{_W / 2:.0f}" y="{_H - 12}" font-family="sans-serif" '
        f'font-size="13" text-anchor="middle">generation</text>'
        f'<text x="16" y="{_H / 2:.0f}" font-family="sans-serif" font-size="13" '
        f'transform="rotate(-90 16 {_H / 2:.0f})" text-anchor="middle">Hill q1 (effective)</text>'
        f'<text x="{_PAD}" y="{_PAD - 18}" font-family="sans-serif" font-size="11" '
        f'fill="#555">y in [{ymin:.2f}, {ymax:.2f}]  resolution={_esc(res.featurizer)}</text>'
    )
    poly = f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{pts}"/>'
    return _frame(title, axes + poly + dots + labels)


_NON_CLAIM = "realized-sample diversity only; does not estimate population diversity"


def _frame(title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}">'
        f'<rect width="{_W}" height="{_H}" fill="white"/>'
        f'<text x="{_PAD}" y="28" font-family="sans-serif" font-size="16" '
        f'font-weight="bold">{_esc(title)}</text>'
        f'<text x="{_PAD}" y="44" font-family="sans-serif" font-size="10" '
        f'fill="#777">{_esc(_NON_CLAIM)}</text>'
        f"{body}</svg>\n"
    )


def write_svg(diag: Diagnosis, path: str | Path) -> str:
    svg = diagnosis_to_svg(diag)
    Path(path).write_text(svg, encoding="utf-8")
    return svg
