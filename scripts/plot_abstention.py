"""Plot the abstention-performance curves for every layer combination against random rejection.

The figure the paper turns on. Each line is one auditor configuration: coverage on the x-axis,
realised mean Sharpe of the retained set on the y-axis. The shaded band is the 95% interval of
random rejection at matched coverage, and a line inside that band is a null result — the honest
majority outcome and the one this lab exists to be able to state.

The title records whether the performance source is the development period (diagnostic) or the
holdout (reportable), because a reader who mistakes one for the other draws the wrong conclusion
from an otherwise identical picture.

Usage:
    python scripts/plot_abstention.py --ablation runs/pooled/ablation_development.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no display on this machine; write straight to file
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.abstention import random_baseline  # noqa: E402

#: Colour-blind-safe qualitative set; seven combinations need seven distinguishable lines.
PALETTE = ("#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161", "#949494", "#d55e00")


def load_performance(
    run_dir: Path, *, holdout: bool, flat_tolerance: float = 1e-9
) -> dict[str, float]:
    """The exact population and values the ablation scored, so the band matches the curves.

    Eligibility is always the development judgment — executed and not flat — because that is how the
    ablation fixes its population. The *values* come from whichever window was scored. Reading
    development values for a holdout plot would draw a random-rejection band from the wrong
    distribution entirely, and the band is the only thing on the figure that says what null looks
    like.
    """
    development = json.loads((run_dir / "backtest_results.json").read_text(encoding="utf-8"))
    eligible = {
        r["name"] for r in development
        if r["outcome"] == "evaluated" and r.get("sharpe") is not None
        and abs(float(r["sharpe"])) >= flat_tolerance
    }
    source = "holdout_results.json" if holdout else "backtest_results.json"
    records = json.loads((run_dir / source).read_text(encoding="utf-8"))
    return {
        r["name"]: float(r["sharpe"])
        for r in records
        if r["outcome"] == "evaluated" and r.get("sharpe") is not None
        and r["name"] in eligible
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.ablation.read_text(encoding="utf-8"))
    run_dir = args.ablation.parent
    performance = load_performance(run_dir, holdout=bool(data["reportable_auap"]))
    _, intervals, auap_interval = random_baseline(performance, seed=42)

    coverages = data["combinations"][0]["curve"]["coverages"]
    lower = [i[0] for i in intervals]
    upper = [i[1] for i in intervals]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(coverages, lower, upper, color="0.85", label="random rejection, 95%")

    for colour, row in zip(PALETTE, sorted(data["combinations"], key=lambda r: -r["auap"]),
                           strict=False):
        ax.plot(
            coverages, row["curve"]["performance"], marker="o", markersize=3.5,
            color=colour, linewidth=1.6,
            label=f"{'+'.join(row['layers'])}  (AUAP {row['auap']:.3f})",
        )

    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle=":")
    ax.set_xlabel("coverage $c$ — fraction of candidates acted upon, most-trusted first")
    ax.set_ylabel("$P(c)$ — mean Sharpe of the retained set")

    source = data["performance_source"]
    tag = "HOLDOUT — reportable" if data["reportable_auap"] else "development — DIAGNOSTIC ONLY"
    ax.set_title(
        f"Abstention–performance curve by auditor configuration\n"
        f"{source} performance, n = {data['n_candidates']} rankable   [{tag}]",
        fontsize=10.5,
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(0.0, 1.02)
    fig.tight_layout()

    out = args.out or run_dir / f"abstention_{source}.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(f"random baseline AUAP 95%: [{auap_interval[0]:.4f}, {auap_interval[1]:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
