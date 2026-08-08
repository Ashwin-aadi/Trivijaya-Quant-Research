"""Render the figures the three papers share, from META.json. Never drawn by hand.

Each figure answers one question the papers ask in words, and none is decorative:

- ``fig_front_vs_back``   Does a more capable generator move executability, or out-of-sample
                          performance? Two panels on a shared generator axis.
- ``fig_compute_matched`` Does any scaffolded generation method beat plain prompting once plain
                          prompting is given the same token budget?
- ``fig_null_forest``     Does the auditor's ranking beat random rejection on any corpus? One row
                          per corpus, each against its own bootstrap interval.
- ``fig_funnel``          Where does a generated corpus actually die?
- ``fig_shortcut_safety`` Which computational shortcuts reproduce the expensive measurement?
- ``fig_fragility_by_origin`` Does fragility depend on who wrote the strategy, or how it was
                          prompted? One shared definition, one shared axis.

Usage:
    python scripts/build_paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

META = Path("benchmarks/generationbench/META.json")
OUTDIR = Path("papers/figures")

NAVY = "#0E2E5C"
STEEL = "#2C6BA8"
SKY = "#5B93C7"
RUST = "#9C4221"
MIST = "#F0F5FB"
GREY = "#9AA5B1"

PARADIGM_LABEL = {
    "G1": "plain", "G2": "chain-of-\nthought", "G4": "planning",
    "G5": "reflection", "G6": "graph-of-\nthoughts", "G7": "MCTS",
}


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=NAVY, labelsize=8.5)
    ax.yaxis.label.set_color(NAVY)
    ax.xaxis.label.set_color(NAVY)


def front_vs_back(meta: dict) -> None:
    """Executability and out-of-sample performance, on one generator axis."""
    gen = meta["generator_axis"]
    labels, trade, hold = [], [], []
    for record in gen:
        name = "local 7B" if record["role"] == "reference" else record["generator"]
        labels.append(name)
        trade.append((record["funnel"]["position_taking_rate"] or 0) * 100)
        hold.append(record["funnel"]["holdout_median_sharpe"] or 0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.3))
    colours = [STEEL] + [NAVY] * (len(labels) - 1)

    left.bar(labels, trade, color=colours, width=0.62)
    left.set_ylabel("executes and takes a position (%)")
    left.set_ylim(0, 108)
    left.set_title("Front of the funnel: the code", color=NAVY, fontsize=10,
                   fontweight="bold", loc="left")
    for i, value in enumerate(trade):
        left.text(i, value + 2.5, f"{value:.1f}", ha="center", color=NAVY, fontsize=8.5)
    _style(left)

    right.bar(labels, hold, color=colours, width=0.62)
    right.axhline(0, color=GREY, lw=0.9)
    right.set_ylabel("median holdout Sharpe")
    right.set_title("Back of the funnel: the alpha", color=NAVY, fontsize=10,
                    fontweight="bold", loc="left")
    for i, value in enumerate(hold):
        offset = -0.11 if value < 0 else 0.05
        right.text(i, value + offset, f"{value:.2f}", ha="center", color=NAVY, fontsize=8.5)
    right.set_ylim(min(hold) - 0.32, max(0.18, max(hold) + 0.18))
    _style(right)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_front_vs_back.png", dpi=220)
    plt.close(fig)


def compute_matched(meta: dict) -> None:
    """Each scaffolded method against plain prompting at the same token budget."""
    records = [r for r in meta["methodology_axis"] if r["arm"] != "G1"]
    labels = [PARADIGM_LABEL[r["arm"]] for r in records]
    control = [r["compute_matched"]["control_yield"] * 100 for r in records]
    treat = [r["compute_matched"]["treatment_yield"] * 100 for r in records]
    ks = [r["compute_matched"]["k"] for r in records]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.bar([i - 0.19 for i in x], control, width=0.36, color=STEEL,
           label="plain prompting at the same budget, best of $k$")
    ax.bar([i + 0.19 for i in x], treat, width=0.36, color=RUST,
           label="the scaffolded method")
    for i, (c, t, k) in enumerate(zip(control, treat, ks, strict=True)):
        ax.text(i - 0.19, c + 1.6, f"{c:.1f}", ha="center", color=NAVY, fontsize=8.5)
        ax.text(i + 0.19, t + 1.6, f"{t:.1f}", ha="center", color=NAVY, fontsize=8.5)
        ax.text(i, -8.5, f"$k={k}$", ha="center", color=GREY, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("yield: draws producing a rankable strategy (%)")
    ax.set_ylim(-12, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=NAVY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_compute_matched.png", dpi=220)
    plt.close(fig)


def null_forest(meta: dict) -> None:
    """One row per corpus: best AUAP against that corpus's own random-rejection interval."""
    rows = []
    for record in meta["methodology_axis"]:
        block = record["ablation_holdout"]
        if block:
            rows.append((f"method: {record['paradigm']}", block))
    for record in meta["generator_axis"]:
        block = record["ablation_holdout"]
        if block and record["role"] == "frontier arm":
            rows.append((f"generator: {record['generator']}", block))
    p1 = json.loads(Path("runs/pooled/ablation_holdout.json").read_text(encoding="utf-8"))
    combos = p1["combinations"]
    rows.append(("AlphaAudit pooled corpus", {
        "best_auap": max(c["auap"] for c in combos),
        "baseline_interval": p1["random_baseline_auap_interval"],
        "n_ranked": p1["n_candidates"]}))

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    for i, (_label, block) in enumerate(rows):
        lo, hi = block["baseline_interval"]
        ax.plot([lo, hi], [i, i], color=SKY, lw=7, alpha=0.55,
                solid_capstyle="butt",
                label="random rejection, 95% interval" if i == 0 else None)
        ax.plot(block["best_auap"], i, "o", color=NAVY, ms=6.5,
                label="best auditor configuration" if i == 0 else None)
        ax.text(0.42, i, f"n={block['n_ranked']}", color=GREY, fontsize=8,
                va="center", transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("AUAP — area under the abstention--performance curve (higher is better)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=NAVY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_null_forest.png", dpi=220)
    plt.close(fig)


def funnel(meta: dict) -> None:
    """Where a generated corpus dies, per generation method, as a share of draws."""
    records = meta["methodology_axis"]
    labels = [PARADIGM_LABEL[r["arm"]] for r in records]
    executes = [r["funnel"]["execution_rate"] * 100 for r in records]
    trades = [r["funnel"]["position_taking_rate"] * 100 for r in records]
    survives = [r["full_stack_survival"] * 100 for r in records]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.bar(x, [100] * len(labels), width=0.66, color=MIST, label="drawn")
    ax.bar(x, executes, width=0.66, color=SKY, label="executes on real data")
    ax.bar(x, trades, width=0.66, color=STEEL, label="takes a position")
    ax.bar(x, survives, width=0.66, color=NAVY, label="clears the whole stack")
    for i, value in enumerate(survives):
        ax.text(i, value + 2.0, f"{value:.1f}", ha="center", color=NAVY, fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("share of draws (%)")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", labelcolor=NAVY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_funnel.png", dpi=220)
    plt.close(fig)


def shortcut_safety() -> None:
    """Which computational shortcuts reproduce the expensive measurement, and which do not.

    Three rank agreements on the same 125 strategies. Two are the cheap stress tier against full
    counterfactual re-simulation -- one on performance, one on fragility. The third is the
    across-regime shortcut validated against its bootstrapped counterpart. Plotted together
    because the paper's claim is that shortcut safety is measured, not assumed, and the three
    outcomes differ.
    """
    tier = json.loads(
        Path("data/processed/tier_comparison.json").read_text(encoding="utf-8"))
    rerun = json.loads(
        Path("data/processed/rerun_decision.json").read_text(encoding="utf-8"))
    agree = tier["tier_agreement"]["conditional"]

    rows = [
        ("Cheap tier reproduces\nmean performance", agree["spearman_mean_sharpe"],
         agree["n_mean"], STEEL),
        ("Cheap tier reproduces\nfragility", agree["spearman_fragility"], agree["n"], RUST),
        ("Free across-regime measure\nreproduces the bootstrapped one",
         rerun["real_vs_bootstrap_spearman"], rerun["n_strategies"], NAVY),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    for i, (_label, rho, n, colour) in enumerate(rows):
        ax.barh(i, rho, color=colour, height=0.55)
        ax.text(rho + 0.012, i, f"{rho:.3f}   n={n}", va="center", color=NAVY, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Spearman rank agreement with the expensive measurement")
    ax.axvline(0.9, color=GREY, lw=0.8, ls="--")
    ax.text(0.9, -0.72, "0.90", ha="center", color=GREY, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_shortcut_safety.png", dpi=220)
    plt.close(fig)


def fragility_by_origin(meta: dict) -> None:
    """Median fragility across regimes, by who wrote the strategy and by how it was prompted.

    One shared axis and one shared definition -- the charter's primary fragility, measured across
    labelled regimes -- so the reader can see at a glance whether the quantity moves with the
    generator or with the generation method. Sample sizes are printed because they range from 12
    to 125 and the small arms must not be read as firmly as the large ones.
    """
    gap = json.loads(
        Path("data/processed/frontier_gap_measures.json").read_text(encoding="utf-8"))

    rows: list[tuple[str, float, int, str]] = [
        ("local 7B (audit corpus)", gap["local_regime_fragility_median"],
         gap["local_n_primary"], STEEL),
    ]
    for arm, label in (("gpt", "GPT"), ("claude", "Claude"), ("gemini", "Gemini")):
        block = gap["arms"][arm]["regime_fragility"]
        rows.append((label, block["median"], block["n"], NAVY))
    for record in meta["methodology_axis"]:
        rows.append((PARADIGM_LABEL[record["arm"]].replace("\n", " "),
                     record["fragility_tier2_across_regimes_median"],
                     record["fragility_n"], SKY))

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    for i, (_label, value, n, colour) in enumerate(rows):
        ax.barh(i, value, color=colour, height=0.6)
        ax.text(value + 0.015, i, f"{value:.3f}   n={n}", va="center", color=NAVY, fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[1] for r in rows) * 1.42)
    ax.set_xlabel("median fragility across labelled regimes (lower is more stable)")
    ax.axhline(3.5, color=GREY, lw=0.8)
    ax.text(0.985, 0.985, "generators", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=GREY, fontweight="bold")
    ax.text(0.985, 0.40, "generation methods", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=GREY, fontweight="bold")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_fragility_by_origin.png", dpi=220)
    plt.close(fig)


def main() -> int:
    configure_logging()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8"))
    front_vs_back(meta)
    compute_matched(meta)
    null_forest(meta)
    funnel(meta)
    shortcut_safety()
    fragility_by_origin(meta)
    _log.info("wrote 6 figures to %s", OUTDIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
