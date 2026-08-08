"""Render the site's figures from frozen artifacts, so nothing on the page is drawn by hand.

The public page states that this lab does not show numbers it did not compute. A schematic curve
illustrating what overfitting *looks like* would quietly contradict that, so the abstention figure
is generated here from ``runs/pooled/ablation_holdout.json`` -- the artifact AlphaAudit's paper
reports -- and injected between markers in ``docs/index.html``.

**What the figure shows.** For each combination of auditor layers, the abstention-performance curve
P(c): realised holdout Sharpe among the fraction ``c`` of candidates the auditor was most confident
in. An informative auditor produces a curve that rises as ``c`` falls, because refusing to act on
what it doubts should leave something better. The shaded band is the random-rejection baseline at
matched coverage. **No combination clears it**, which is the published result and the reason the
figure is worth showing rather than a picture of a strategy that worked.

Re-run after any change to the ablation artifact. Idempotent: it replaces the marked region rather
than appending to it.

Usage:
    python scripts/build_site_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

ABLATION = Path("runs/pooled/ablation_holdout.json")
PAGE = Path("docs/index.html")
BEGIN, END = "<!-- FIGURE:AUAP:BEGIN -->", "<!-- FIGURE:AUAP:END -->"

# Plot geometry, in the SVG's own user units.
W, H = 1000, 380
L, R, T, B = 74, 24, 22, 52


def _scales(combos: list[dict[str, Any]]) -> tuple[float, float]:
    """Y bounds covering every plotted point, padded so no curve touches the frame."""
    values = [v for c in combos for v in c["curve"]["performance"]]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _point(coverage: float, value: float, lo: float, hi: float) -> tuple[float, float]:
    """Map (coverage, performance) to SVG coordinates.

    Coverage runs 1.0 -> 0.05 left to right, so moving right means the auditor is being *more*
    selective. That direction is deliberate: the claim under test is that performance improves as
    coverage falls, and a reader should see the claimed effect as a rise to the right.
    """
    x = L + (1.0 - coverage) / (1.0 - 0.05) * (W - L - R)
    y = T + (hi - value) / (hi - lo) * (H - T - B)
    return x, y


def _path(curve: dict[str, list[float]], lo: float, hi: float) -> str:
    pts = [_point(c, v, lo, hi)
           for c, v in zip(curve["coverages"], curve["performance"], strict=True)]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def _svg(payload: dict[str, Any]) -> str:
    """The abstention-performance figure, with the random baseline drawn as a band."""
    combos = payload["combinations"]
    lo, hi = _scales(combos)
    best = max(combos, key=lambda c: c["auap"])
    b_lo, b_hi = payload["random_baseline_auap_interval"]

    band_top = _point(1.0, b_hi, lo, hi)[1]
    band_bottom = _point(1.0, b_lo, lo, hi)[1]

    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Abstention-performance curves for '
        'every auditor layer combination, none of which beats random rejection">',
        f'<rect x="{L}" y="{band_top:.1f}" width="{W - L - R}" '
        f'height="{band_bottom - band_top:.1f}" fill="rgba(159,182,220,.16)"/>',
        f'<text x="{L + 10}" y="{band_top - 7:.1f}" class="ann">random rejection, matched '
        'coverage</text>',
    ]

    # Gridlines at round Sharpe values inside the range.
    tick = -0.5
    while tick > lo:
        if tick < hi:
            y = _point(1.0, tick, lo, hi)[1]
            parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{L - 10}" y="{y + 4:.1f}" class="ytick">{tick:.1f}</text>')
        tick -= 0.5

    for combo in combos:
        klass = "curve best" if combo is best else "curve"
        parts.append(f'<path d="{_path(combo["curve"], lo, hi)}" class="{klass}"/>')

    parts.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" class="axis"/>')
    for coverage in (1.0, 0.8, 0.6, 0.4, 0.2, 0.05):
        x = _point(coverage, hi, lo, hi)[0]
        parts.append(f'<text x="{x:.1f}" y="{H - B + 22}" class="xtick">{coverage:g}</text>')
    parts.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" class="xlab">'
                 'Coverage — fraction of candidates acted upon</text>')
    parts.append(f'<text x="{L - 56}" y="{(T + H - B) / 2:.0f}" class="ylab" '
                 f'transform="rotate(-90 {L - 56} {(T + H - B) / 2:.0f})">Holdout Sharpe</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _table(payload: dict[str, Any]) -> str:
    """Every combination's AUAP, in the order the ablation reports them."""
    rows = []
    for combo in payload["combinations"]:
        layers = " + ".join(combo["layers"])
        verdict = "beats random" if combo["beats_random"] else "no"
        rows.append(
            f'<tr><td>{layers}</td><td class="n">{combo["auap"]:.4f}</td>'
            f'<td class="n">{combo["p_at_005"]:.4f}</td><td class="v">{verdict}</td></tr>'
        )
    lo, hi = payload["random_baseline_auap_interval"]
    return (
        '<table class="abl"><thead><tr><th>Auditor layers</th><th class="n">AUAP</th>'
        '<th class="n">P at 5% coverage</th><th class="v">Beats random</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'<p class="fcap" style="margin-top:22px">Random-rejection baseline AUAP: '
        f'<b>{lo:.4f} to {hi:.4f}</b>, bootstrap interval, n = {payload["n_candidates"]} ranked '
        'candidates on the 2025 holdout. Every combination falls inside or below it. '
        'The abstention frontier is the metric this benchmark was built to produce, and it '
        'returned a null.</p>'
    )


def main() -> int:
    configure_logging()
    payload = json.loads(ABLATION.read_text(encoding="utf-8"))
    if not payload.get("reportable_auap"):
        _log.error("%s is not marked reportable; refusing to publish it", ABLATION)
        return 1

    block = f"{BEGIN}\n{_svg(payload)}\n{_table(payload)}\n{END}"
    page = PAGE.read_text(encoding="utf-8")
    if BEGIN not in page or END not in page:
        _log.error("markers not found in %s", PAGE)
        return 1
    head, rest = page.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    PAGE.write_text(head + block + tail, encoding="utf-8")

    beats = sum(1 for c in payload["combinations"] if c["beats_random"])
    _log.info("rendered %d combinations, %d beating random, into %s",
              len(payload["combinations"]), beats, PAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
