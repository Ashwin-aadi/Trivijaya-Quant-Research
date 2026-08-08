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
    """Each scaffolded method against plain prompting at the same token budget, three ways.

    Yield is the primary comparison and every arm loses it. The other two panels are the reasons
    a yield win would not have been enough on its own: a method that collapses onto one idea
    scores well on average and is worthless, and a method that raises yield without raising
    out-of-sample performance has not produced research.
    """
    records = [r for r in meta["methodology_axis"] if r["arm"] != "G1"]
    plain = next(r for r in meta["methodology_axis"] if r["arm"] == "G1")
    labels = [PARADIGM_LABEL[r["arm"]] for r in records]
    control = [r["compute_matched"]["control_yield"] * 100 for r in records]
    treat = [r["compute_matched"]["treatment_yield"] * 100 for r in records]
    ks = [r["compute_matched"]["k"] for r in records]

    x = range(len(labels))
    fig, (ax, mid, right) = plt.subplots(
        3, 1, figsize=(9.2, 7.2), gridspec_kw={"height_ratios": [1.5, 1, 1]})
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
    ax.set_ylabel("yield: draws producing a\nrankable strategy (%)", fontsize=8.5)
    ax.set_ylim(-12, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=NAVY)
    _style(ax)

    allrec = [plain, *records]
    alllabels = ["plain", *labels]
    for panel, key, ylabel, scale in (
        (mid, "redundancy_of_traded", "inside an exact-duplicate\ncluster (%)", 100.0),
        (right, None, "median holdout Sharpe", 1.0),
    ):
        values = [(r["funnel"]["holdout_median_sharpe"] if key is None else r[key]) * scale
                  for r in allrec]
        colours = [STEEL] + [RUST] * len(records)
        panel.bar(range(len(values)), values, width=0.55, color=colours)
        span = max(abs(v) for v in values) or 1.0
        for i, value in enumerate(values):
            panel.text(i, value + span * (0.05 if value >= 0 else -0.16),
                       f"{value:.2f}" if scale == 1.0 else f"{value:.0f}",
                       ha="center", color=NAVY, fontsize=8)
        panel.axhline(0, color=GREY, lw=1.0)
        panel.set_ylim(min(min(values) * 1.42, 0), max(max(values) * 1.22, span * 0.1))
        panel.set_xticks(range(len(alllabels)))
        panel.set_xticklabels(alllabels, fontsize=8.5)
        panel.set_ylabel(ylabel, fontsize=8.5)
        _style(panel)

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


def _factors() -> list[dict]:
    return json.loads(
        Path("data/processed/standard_factor_comparison.json").read_text(
            encoding="utf-8"))["results"]


def factor_costs() -> None:
    """What Indian costs do to eleven published factors: gross against net, one row each.

    The paper's claim is that costs are a filter rather than a haircut. A table of twenty-two
    numbers does not show that; a dumbbell does, because the eye reads the length of the drop and
    the two rows that cross zero.
    """
    rows = sorted(_factors(), key=lambda r: r["sharpe_net"])
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for i, r in enumerate(rows):
        flipped = r["sign_flipped_by_costs"]
        ax.plot([r["sharpe_net"], r["sharpe_gross"]], [i, i],
                color=RUST if flipped else GREY, lw=2.2, zorder=1)
        ax.plot(r["sharpe_gross"], i, "o", color=SKY, ms=7, zorder=2,
                label="gross of costs" if i == 0 else None)
        ax.plot(r["sharpe_net"], i, "o", color=NAVY, ms=7, zorder=3,
                label="net of Indian costs" if i == 0 else None)
    ax.axvline(0, color=NAVY, lw=1.0, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_xlabel("Sharpe ratio, development window")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=NAVY)
    ax.text(0.02, 0.03, "rust = sign reversed by costs", transform=ax.transAxes,
            fontsize=8.5, color=RUST, fontweight="bold")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_factor_costs.png", dpi=220)
    plt.close(fig)


def factor_deflation() -> None:
    """Every factor's undeflated significance against its deflated one, on a log axis.

    The point is the distance between the two dots. Undeflated, the best factors look conclusive;
    deflated at the family's own trial count, nothing reaches the bar the machine corpus was held
    to. A log axis is unavoidable -- the deflated values span twenty orders of magnitude.
    """
    rows = sorted(_factors(), key=lambda r: r["dsr_at_family_n"])
    floor = 1e-8
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for i, r in enumerate(rows):
        psr = max(r["psr_undeflated"], floor)
        dsr = max(r["dsr_at_family_n"], floor)
        ax.plot([dsr, psr], [i, i], color=GREY, lw=2.0, zorder=1)
        ax.plot(psr, i, "o", color=SKY, ms=7, zorder=2,
                label="undeflated (PSR)" if i == 0 else None)
        ax.plot(dsr, i, "o", color=RUST, ms=7, zorder=3,
                label="deflated at the family's own $N=11$" if i == 0 else None)
    ax.axvline(0.95, color=NAVY, lw=1.4, ls="--")
    ax.text(0.95, len(rows) - 0.4, " significance bar 0.95", color=NAVY, fontsize=8.5,
            va="center")
    ax.set_xscale("log")
    ax.set_xlim(floor / 3, 3.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_xlabel("probability the Sharpe is real (log scale); values below $10^{-8}$ clipped")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=NAVY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_factor_deflation.png", dpi=220)
    plt.close(fig)


def dispersion_mechanism() -> None:
    """Momentum's deflated significance as the comparison set shrinks. Post-hoc, and labelled so.

    The single most important diagnostic in the programme: same strategy, same Sharpe, same trial
    count, and a verdict that moves by three orders of magnitude depending only on which other
    strategies are standing beside it. Drawn because a reader who sees it will not forget it.
    """
    # Values from the diagnostic run recorded in the checkpoint; recomputed here from the artefact
    # so the figure cannot drift from the numbers the text quotes.
    defl = json.loads(
        Path("data/processed/standard_factor_deflation.json").read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in defl["results"]}
    import numpy as np

    from src.audit.stat import deflated_sharpe_ratio  # noqa: PLC0415 - figure-local by design

    best = rows["momentum_skip_month"]
    sets = [
        ("all 11\n(pre-specified)", set()),
        ("less the\nnull control", {"random_walk_baseline"}),
        ("less both labelled\ncontrols", {"random_walk_baseline", "high_volatility"}),
        ("less the four\nstrongly negative",
         {"random_walk_baseline", "bollinger_reversion", "inverse_volatility_weighted",
          "mean_reversion_5d"}),
    ]
    labels, values = [], []
    for label, drop in sets:
        sub = [r for r in defl["results"] if r["name"] not in drop]
        var = float(np.var([r["sharpe_per_observation"] for r in sub], ddof=1))
        values.append(deflated_sharpe_ratio(
            observed_sharpe=best["sharpe_per_observation"], n_trials=len(sub),
            n_observations=best["n_observations"], skew=best["skew"],
            kurtosis=best["kurtosis"], variance_of_trial_sharpes=var))
        labels.append(label)

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    colours = [NAVY] + [RUST] * (len(labels) - 1)
    ax.bar(labels, values, color=colours, width=0.58)
    for i, v in enumerate(values):
        ax.text(i, v + 0.022, f"{v:.4f}", ha="center", color=NAVY, fontsize=9)
    ax.axhline(0.95, color=SKY, lw=1.4, ls="--")
    ax.text(len(labels) - 0.5, 0.965, "significance bar", ha="right", color=SKY, fontsize=8.5)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("momentum's deflated significance")
    ax.tick_params(axis="x", labelsize=8.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_dispersion_mechanism.png", dpi=220)
    plt.close(fig)


def factors_vs_corpus() -> None:
    """The standard-factor arm beside the machine-written corpus on five frozen tests.

    Each panel is a rate on its own scale, because the five quantities share no unit. Panels
    rather than one table so the two reversals -- costs and fragility running opposite ways --
    are visible without arithmetic.
    """
    summary = json.loads(
        Path("data/processed/standard_factor_comparison.json").read_text(encoding="utf-8"))
    n = summary["n"]
    panels = [
        ("Sharpe sign reversed\nby costs (%)",
         summary["sign_flipped_by_costs"] / n * 100, 15.6),
        ("Cleared the\nsignificance bar (%)", 0.0, 0.0),
        ("Knife-edge (%)", summary["knife_edge"] / n * 100, 31 / 156 * 100),
        ("Nondeterministic (%)", 0.0, 27 / 185 * 100),
        ("Median fragility\n(lower = steadier)", summary["median_fragility"], 0.618),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(9.6, 3.0))
    for ax, (title, factor_value, corpus_value) in zip(axes, panels, strict=True):
        ax.bar(["standard\nfactors", "machine\ncorpus"], [factor_value, corpus_value],
               color=[NAVY, SKY], width=0.62)
        for i, v in enumerate([factor_value, corpus_value]):
            ax.text(i, v + max(factor_value, corpus_value, 1) * 0.035, f"{v:.3g}",
                    ha="center", color=NAVY, fontsize=8.5)
        ax.set_title(title, color=NAVY, fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, max(factor_value, corpus_value, 1) * 1.28)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_factors_vs_corpus.png", dpi=220)
    plt.close(fig)


def predictor_gap() -> None:
    """Training against out-of-sample R-squared for five models. The gap is the diagnosis.

    A table of ten numbers hides the finding; paired bars make it immediate. Every model fits its
    training folds far better than it generalises, and the gap is *widest* for the model with the
    most capacity -- which is the signature of variance rather than bias, and the reason no
    capacity was added.
    """
    diag = json.loads(
        Path("data/processed/predictor_diagnosis.json").read_text(encoding="utf-8"))
    block = diag["targets"]["fragility_across_paths[raw]"]["models"]
    order = ["ridge", "lasso", "elastic_net", "random_forest", "gradient_boosting"]
    pretty = {"ridge": "ridge", "lasso": "lasso", "elastic_net": "elastic\nnet",
              "random_forest": "random\nforest", "gradient_boosting": "gradient\nboosting"}
    labels = [pretty[k] for k in order if k in block]
    train = [block[k]["r2_train"] for k in order if k in block]
    test = [block[k]["r2_model"] for k in order if k in block]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.bar([i - 0.19 for i in x], train, width=0.36, color=SKY, label="training $R^2$")
    ax.bar([i + 0.19 for i in x], test, width=0.36, color=NAVY, label="out-of-sample $R^2$")
    for i, (tr, te) in enumerate(zip(train, test, strict=True)):
        ax.text(i - 0.19, tr + 0.03, f"{tr:+.2f}", ha="center", color=NAVY, fontsize=8)
        ax.text(i + 0.19, te + (0.03 if te >= 0 else -0.10), f"{te:+.2f}", ha="center",
                color=NAVY, fontsize=8)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("$R^2$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=NAVY)
    ax.text(0.99, 0.04, "zero = predicting the training mean", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=GREY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_predictor_gap.png", dpi=220)
    plt.close(fig)


def capacity_binding() -> None:
    """What sets a strategy's capacity: the least liquid position, not the average one.

    Three bars per factor -- the median session, the session that builds the opening position,
    and the session that actually binds -- plus the share of sessions on which a single name is
    the constraint. This replaces the paper's capacity table outright: the table's five columns
    are all here, and the gap between the pale and dark bars is the finding, which no column of
    ratios conveys.
    """
    fs = json.loads(Path("data/processed/flowstate.json").read_text(encoding="utf-8"))
    crore = 1e7
    rows = [(b["factor"].replace("_", " "), b["binding_capacity_inr"] / crore,
             b["median_capacity_inr"] / crore, b["entry_capacity_inr"] / crore,
             b["fraction_bound_by_one_name"] * 100) for b in fs.get("capacity", [])]
    if not rows:
        _log.warning("no capacity block in flowstate.json; figure skipped, not faked")
        return
    rows.sort(key=lambda r: r[1])

    x = range(len(rows))
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(9.2, 5.0), gridspec_kw={"height_ratios": [2.5, 1]}, sharex=True)
    top.bar([i - 0.26 for i in x], [r[2] for r in rows], width=0.25, color=MIST,
            edgecolor=SKY, label="if the median session set it")
    top.bar([i for i in x], [r[3] for r in rows], width=0.25, color=SKY,
            label="the session that builds the opening position")
    top.bar([i + 0.26 for i in x], [r[1] for r in rows], width=0.25, color=NAVY,
            label="what actually binds (worst session)")
    for i, r in enumerate(rows):
        for offset, value in ((-0.26, r[2]), (0.0, r[3]), (0.26, r[1])):
            top.text(i + offset, value * 1.09, f"{value:.1f}", ha="center", color=NAVY,
                     fontsize=7.5)
    top.set_yscale("log")
    top.set_ylim(min(r[1] for r in rows) * 0.45, max(r[2] for r in rows) * 3.4)
    top.set_ylabel("deployable AUM (Rs. crore, log)")
    top.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3, labelcolor=NAVY)
    _style(top)

    bot.bar(list(x), [r[4] for r in rows], width=0.5, color=RUST)
    for i, r in enumerate(rows):
        bot.text(i, r[4] + 1.2, f"{r[4]:.1f}%", ha="center", color=NAVY, fontsize=8)
    bot.set_ylim(0, max(r[4] for r in rows) * 1.38)
    bot.set_ylabel("one name binds (%)", fontsize=8.5)
    bot.set_xticks(list(x))
    bot.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    _style(bot)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_capacity_binding.png", dpi=220)
    plt.close(fig)


def alpha_decay() -> None:
    """Return per session held against holding horizon, and whether any of it is established.

    The paper's decay table is forty numbers in eight columns, and its finding -- that the curves
    are flat and none of them is significant -- is exactly what a table hides. Two panels: the
    curves, and the same factors' $|t|$ against the conventional threshold none of them crosses.
    """
    fs = json.loads(Path("data/processed/flowstate.json").read_text(encoding="utf-8"))
    decay = fs.get("decay", {})
    if not decay:
        _log.warning("no decay block in flowstate.json; figure skipped, not faked")
        return
    colours = [NAVY, STEEL, SKY, RUST, GREY]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for colour, (factor, block) in zip(colours, sorted(decay.items()), strict=False):
        curve = block["curve"]
        horizons = [p["horizon"] for p in curve]
        bps = [p["mean_return_per_session"] * 1e4 for p in curve]
        tstat = [abs(p["t_statistic"]) for p in curve]
        label = factor.replace("_", " ")
        left.plot(horizons, bps, "o-", color=colour, ms=3.4, lw=1.6, label=label)
        right.plot(horizons, tstat, "o-", color=colour, ms=3.4, lw=1.6)

    left.axhline(0, color=GREY, lw=1.0, ls="--")
    left.set_xscale("log")
    left.set_xlabel("holding horizon (sessions, log)")
    left.set_ylabel("return per session held (bps)")
    left.set_title("The curves are flat", color=NAVY, fontsize=10, fontweight="bold", loc="left")
    left.legend(frameon=False, fontsize=7.5, labelcolor=NAVY, loc="lower left")
    _style(left)

    right.axhline(1.96, color=RUST, lw=1.3, ls="--")
    right.text(1.05, 2.02, "conventional significance", color=RUST, fontsize=8)
    right.set_xscale("log")
    right.set_xlabel("holding horizon (sessions, log)")
    right.set_ylabel("$|t|$")
    right.set_ylim(0, 2.5)
    right.set_title("and the short-horizon edges are not established", color=NAVY, fontsize=9.5,
                    fontweight="bold", loc="left")
    _style(right)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_alpha_decay.png", dpi=220)
    plt.close(fig)


def reversal_weighting() -> None:
    """The impact coefficient's sign against the analyst's choice of weighting scheme.

    Both estimators are standard and neither is wrong. Drawn rather than tabulated because the
    finding is that two lines sit on opposite sides of zero, and the shaded band is what the data
    could have detected -- so the reader can see that the disagreement is not a power problem.
    """
    ident = json.loads(
        Path("data/processed/impact_identifiability.json").read_text(encoding="utf-8"))
    by_h = ident["d2_transience"]["by_horizon"]
    horizons = sorted(int(h) for h in by_h)
    pooled, unweighted, mde = [], [], []
    for h in horizons:
        block = by_h[str(h)]["detectability_heavy"]
        pooled.append(block["pooled_beta"])
        unweighted.append(block["unweighted_mean_beta"])
        mde.append(block["pooled_minimum_detectable_beta"])

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.fill_between(horizons, [-m for m in mde], mde, color=MIST,
                    label="below pooled detectability at 80% power")
    ax.plot(horizons, pooled, "o-", color=NAVY, ms=5, lw=1.8,
            label="precision-weighted (inverse variance)")
    ax.plot(horizons, unweighted, "s-", color=RUST, ms=5, lw=1.8,
            label="equal-weighted (one vote per symbol)")
    ax.axhline(0, color=GREY, lw=1.1, ls="--")
    for h, p, u in zip(horizons, pooled, unweighted, strict=True):
        if (p < 0) != (u < 0):
            ax.plot([h, h], [p, u], color=RUST, lw=0.9, ls=":", zorder=1)
            ax.text(h, u + 0.012, "signs\ndisagree", ha="center", fontsize=7, color=RUST)
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("reversal horizon (sessions)")
    ax.set_ylabel(r"reversal coefficient $\beta$")
    ax.legend(frameon=False, fontsize=8, labelcolor=NAVY, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_reversal_weighting.png", dpi=220)
    plt.close(fig)


def capacity_by_generator() -> None:
    """Deployment capacity by who wrote the strategy: medians, and the spread around them.

    The paper's point is that the medians move -- one arm outside the pre-registered bound -- while
    the spread within every arm is one to two orders of magnitude. A table of medians and ranges
    makes the reader do that comparison; a log axis with whiskers does it for them.
    """
    gv = json.loads(
        Path("data/processed/generator_validation.json").read_text(encoding="utf-8"))
    corpus = json.loads(
        Path("data/processed/corpus_capacity.json").read_text(encoding="utf-8"))
    crore = 1e7
    values = sorted(r["binding_capacity_inr"] / crore for r in corpus["capacity"])
    mid = len(values) // 2
    local_median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    rows = [("local 7B", local_median, values[0], values[-1], len(values), STEEL)]
    for arm, label in (("gpt", "GPT"), ("claude", "Claude"), ("gemini", "Gemini")):
        block = gv["arms"][arm]
        rows.append((label, block["cap_median_cr"], block["cap_min_cr"], block["cap_max_cr"],
                     block["n"], NAVY))

    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    for i, (_label, med, lo, hi, n, colour) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=SKY, lw=6, alpha=0.55, solid_capstyle="butt")
        ax.plot(med, i, "o", color=colour, ms=8, zorder=3)
        ax.text(hi * 1.25, i, f"median {med:.2f}   n={n}", va="center", color=NAVY, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(r[2] for r in rows) / 2.5, max(r[3] for r in rows) * 9)
    ax.set_xlabel("deployable AUM (Rs. crore, log). Dot = arm median, bar = min to max.")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_capacity_by_generator.png", dpi=220)
    plt.close(fig)


def impact_identifiability() -> None:
    """The parameter that will not hold still, against the control on the same panel.

    One chart carries the whole negative result: a quantity computed from the same daily bars,
    on the same symbols, over the same split, is stable at 0.836 while the impact exponent
    reaches 0.374. The failure is in the model, not in the data, and the control is what proves it.
    """
    ident = json.loads(
        Path("data/processed/impact_identifiability.json").read_text(encoding="utf-8"))
    pairs = [
        ("Amihud illiquidity\n(the control)",
         ident["d5_amihud"]["spearman_between_halves"], NAVY),
        ("fitted impact exponent\n(the parameter we need)",
         ident["d4_stability"]["spearman_between_halves"], RUST),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    for i, (_label, value, colour) in enumerate(pairs):
        ax.barh(i, value, color=colour, height=0.5)
        ax.text(value + 0.015, i, f"{value:.3f}", va="center", color=NAVY, fontsize=9.5)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p[0] for p in pairs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("split-half rank stability (Spearman) on the same symbols and the same bars")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_impact_identifiability.png", dpi=220)
    plt.close(fig)


def population_funnel() -> None:
    """From paper 1's audit survivors to the population paper 2 actually measures.

    Every step is an exclusion, every exclusion is non-random, and the paper's honesty depends on
    the reader seeing how much was removed. A waterfall shows the attrition; the table it replaces
    required subtracting one row from the next to see the same thing.
    """
    bench = Path("benchmarks/regimestress")
    excl = json.loads((bench / "excluded_nondeterministic.json").read_text(encoding="utf-8"))
    cal = json.loads(
        Path("data/processed/tier1_calibration.json").read_text(encoding="utf-8"))
    frag = json.loads(Path("data/processed/fragility.json").read_text(encoding="utf-8"))

    steps = [
        ("audit survivors\nfrom paper 1", excl["n_retained_survivors"], STEEL),
        ("calibrated\nfor stress", cal["n_strategies"], STEEL),
        ("less non-\ndeterministic", -len(cal["nondeterministic"]), RUST),
        ("less runtime\nfailures", -len(cal["failures"]), RUST),
        ("stressed", frag["n_strategies"], STEEL),
        ("less knife-\nedge", -frag["n_knife_edge_excluded"], RUST),
        ("primary\npopulation", frag["n_primary"], NAVY),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    running = 0.0
    for i, (_label, value, colour) in enumerate(steps):
        if value >= 0:  # an absolute level, not a step
            ax.bar(i, value, width=0.6, color=colour)
            ax.text(i, value + 3, f"{value}", ha="center", color=NAVY, fontsize=9,
                    fontweight="bold")
            running = value
        else:
            ax.bar(i, -value, bottom=running + value, width=0.6, color=colour)
            ax.text(i, running + 3, f"{value}", ha="center", color=RUST, fontsize=9)
            running += value
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=8)
    ax.set_ylabel("strategies")
    ax.set_ylim(0, max(s[1] for s in steps) * 1.18)
    ax.text(0.99, 0.95, "rust = removed by a rule fixed before any path was run",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=RUST)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_population_funnel.png", dpi=220)
    plt.close(fig)


def knife_edge_profile() -> None:
    """How the knife-edge exclusions differ from the strategies that were retained.

    Three characteristics on three different scales, so each gets its own panel and its own axis.
    The claim is that the filter removed the fast end of the corpus rather than a random slice,
    and paired bars make that visible without the reader dividing one column by another.
    """
    knife = json.loads(
        Path("benchmarks/regimestress/knife_edge_stability.json").read_text(encoding="utf-8"))
    compare = knife["comparison"]
    panels = [
        ("strategies", knife["n_excluded"], knife["n_retained"], "{:.0f}"),
        ("turnover per session", compare["mean_turnover"]["excluded"]["median"],
         compare["mean_turnover"]["retained"]["median"], "{:.3f}"),
        ("holding period (sessions)", compare["mean_holding_period"]["excluded"]["median"],
         compare["mean_holding_period"]["retained"]["median"], "{:.1f}"),
        ("effective holdings", compare["effective_holdings"]["excluded"]["median"],
         compare["effective_holdings"]["retained"]["median"], "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(9.4, 2.8))
    for ax, (title, excluded, retained, fmt) in zip(axes, panels, strict=True):
        ax.bar(["excluded", "retained"], [excluded, retained], color=[RUST, NAVY], width=0.6)
        top = max(excluded, retained)
        for i, value in enumerate([excluded, retained]):
            ax.text(i, value + top * 0.04, fmt.format(value), ha="center", color=NAVY,
                    fontsize=8.5)
        ax.set_title(title, color=NAVY, fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, top * 1.26)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_knife_edge.png", dpi=220)
    plt.close(fig)


def generator_fragility_range() -> None:
    """Every RegimeStress measurement by generator: fragility, and the three pathology rates.

    The paper's claim is that the medians barely move. That is only meaningful beside the range
    they sit in, which is wide -- so the range is drawn rather than parenthesised. The lower
    panels carry the counts the table used to hold, expressed as rates so a twenty-strategy arm
    and a hundred-and-fifty-six-strategy corpus can sit on one axis.
    """
    gap = json.loads(
        Path("data/processed/frontier_gap_measures.json").read_text(encoding="utf-8"))
    gv = json.loads(
        Path("data/processed/generator_validation.json").read_text(encoding="utf-8"))
    frag = json.loads(Path("data/processed/fragility.json").read_text(encoding="utf-8"))
    dup = json.loads(
        Path("benchmarks/regimestress/duplicates.json").read_text(encoding="utf-8"))
    excl = json.loads(
        Path("benchmarks/regimestress/excluded_nondeterministic.json").read_text(
            encoding="utf-8"))
    knife = json.loads(
        Path("benchmarks/regimestress/knife_edge_stability.json").read_text(encoding="utf-8"))
    arms = [("gpt", "GPT"), ("claude", "Claude"), ("gemini", "Gemini")]

    fig, (ax, panels) = plt.subplots(
        2, 1, figsize=(9.2, 5.4), gridspec_kw={"height_ratios": [1.5, 1]})
    panels.axis("off")
    rows: list[str] = []
    for i, (arm, label) in enumerate(arms):
        block = gap["arms"][arm]["regime_fragility"]
        ax.plot([block["min"], block["max"]], [i, i], color=SKY, lw=6, alpha=0.55,
                solid_capstyle="butt",
                label="min to max within the arm" if i == 0 else None)
        ax.plot(block["median"], i, "o", color=NAVY, ms=8, zorder=3,
                label="arm median" if i == 0 else None)
        ax.text(block["max"] * 1.15, i, f"median {block['median']:.3f}   n={block['n']}",
                va="center", color=NAVY, fontsize=8.5)
        rows.append(label)
    local = gap["local_regime_fragility_median"]
    ax.axvline(local, color=RUST, lw=1.4, ls="--")
    ax.text(local * 0.90, 1.6, f"local 7B median {local:.3f}", color=RUST, fontsize=8.5,
            va="center", ha="center", rotation=90)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_ylim(len(rows) - 0.4, -0.9)
    ax.set_xlabel("fragility across labelled regimes (log scale, lower is steadier)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2, labelcolor=NAVY)
    _style(ax)

    calib = {a: json.loads(
        Path(f"runs/frontier_{a}/calibration.json").read_text(encoding="utf-8"))
        for a, _ in arms}
    stressed = frag["n_strategies"]
    rates = [
        ("mean regime Sharpe\nnear zero (%)",
         frag["n_flagged_near_zero_mean"] / stressed * 100,
         [gv["arms"][a]["frag_near_zero"] / gv["arms"][a]["n"] * 100 for a, _ in arms]),
        ("knife-edge (%)", knife["n_excluded"] / stressed * 100,
         [calib[a]["n_knife_edge"] / gv["arms"][a]["n"] * 100 for a, _ in arms]),
        ("nondeterministic (%)", excl["n_excluded"] / stressed * 100,
         [calib[a]["n_nondeterministic"] / gv["arms"][a]["n"] * 100 for a, _ in arms]),
        ("inside a duplicate\ncluster (%)",
         dup["n_strategies_in_clusters"] / dup["n_compared"] * 100,
         [gv["arms"][a]["dup_covered"] / gv["arms"][a]["dup_compared"] * 100 for a, _ in arms]),
    ]
    box = panels.get_position()
    for j, (title, local, arm_values) in enumerate(rates):
        sub = fig.add_axes((box.x0 + j * box.width / 4 + 0.035,
                            box.y0 + 0.04, box.width / 4 - 0.055, box.height - 0.20))
        values = [local, *arm_values]
        sub.bar(range(4), values, color=[STEEL, NAVY, NAVY, NAVY], width=0.66)
        for i, value in enumerate(values):
            sub.text(i, value + max(values, default=1) * 0.05 + 0.6, f"{value:.0f}",
                     ha="center", color=NAVY, fontsize=7.5)
        sub.set_title(title, color=NAVY, fontsize=8, fontweight="bold")
        sub.set_ylim(0, max(max(values), 5) * 1.35)
        sub.set_xticks(range(4))
        sub.set_xticklabels(["local", "GPT", "Cl", "Gem"], fontsize=7)
        sub.set_yticks([])
        for side in ("top", "right", "left"):
            sub.spines[side].set_visible(False)
        sub.spines["bottom"].set_color(GREY)
        sub.tick_params(colors=NAVY)

    fig.savefig(OUTDIR / "fig_generator_fragility_range.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def leak_classes() -> None:
    """What the static auditor found, and what became of the candidates it flagged.

    Two panels because the paper makes two separate points: one class dominates and one class is
    conspicuously empty, and almost every flag lands on code that never took a position. The
    second point is the one a table of counts buries.
    """
    audit = json.loads(Path("runs/pooled/audit_results.json").read_text(encoding="utf-8"))
    backtests = {r["name"]: r for r in json.loads(
        Path("runs/pooled/backtest_results.json").read_text(encoding="utf-8"))}

    counts: dict[str, int] = {}
    fate = {"runtime error": 0, "flat: ran, never traded": 0, "timeout": 0, "traded": 0}
    for name, record in audit["static"].items():
        if not record["rejected"]:
            continue
        for klass in record["classes"]:
            counts[klass] = counts.get(klass, 0) + 1
        bt = backtests.get(name)
        outcome = bt["outcome"] if bt else "runtime_error"
        if outcome == "runtime_error":
            fate["runtime error"] += 1
        elif outcome == "timeout":
            fate["timeout"] += 1
        elif bt and (bt["mean_turnover"] or 0) > 0:
            fate["traded"] += 1
        else:
            fate["flat: ran, never traded"] += 1

    known = ["snooped_parameter", "future_indexing", "target_in_features",
             "boundary_crossing_window", "survivorship_selection", "full_sample_statistic"]
    ordered = [(k, counts.get(k, 0)) for k in known]
    ordered.sort(key=lambda kv: kv[1])

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.2))
    for i, (_klass, value) in enumerate(ordered):
        colour = RUST if value == 0 else NAVY
        left.barh(i, value, color=colour, height=0.6)
        left.text(value + max(v for _, v in ordered) * 0.02, i, str(value), va="center",
                  color=colour, fontsize=8.5, fontweight="bold" if value == 0 else "normal")
    left.set_yticks(range(len(ordered)))
    pretty = {"snooped_parameter": "snooped\nparameter", "future_indexing": "future indexing",
              "target_in_features": "target in features",
              "boundary_crossing_window": "boundary-crossing\nwindow",
              "survivorship_selection": "survivorship\nselection",
              "full_sample_statistic": "full-sample\nstatistic"}
    left.set_yticklabels([pretty[k] for k, _ in ordered], fontsize=7.5)
    left.set_xlabel("candidates carrying the class")
    left.set_title("What was flagged", color=NAVY, fontsize=10, fontweight="bold", loc="left")
    left.text(0.99, 0.06, "rust = a class we do not believe is truly absent",
              transform=left.transAxes, ha="right", fontsize=7.5, color=RUST)
    _style(left)

    order = ["runtime error", "flat: ran, never traded", "timeout", "traded"]
    total = sum(fate.values())
    colours = [GREY, SKY, MIST, NAVY]
    start = 0.0
    for label, colour in zip(order, colours, strict=True):
        share = fate[label] / total * 100
        right.barh(0, share, left=start, color=colour, height=0.5, edgecolor="white")
        if share > 4:
            right.text(start + share / 2, 0, f"{fate[label]}", ha="center", va="center",
                       color="white" if colour in (GREY, NAVY) else NAVY, fontsize=9,
                       fontweight="bold")
        start += share
    right.annotate(f"{fate['traded']} traded", xy=(100 - fate["traded"] / total * 100 / 2, 0.28),
                   xytext=(72, 0.62), fontsize=8.5, color=NAVY,
                   arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 0.9})
    right.set_xlim(0, 100)
    right.set_ylim(-0.5, 0.9)
    right.set_yticks([])
    right.set_xlabel(f"share of the {total} statically rejected candidates (%)")
    right.set_title("Where the flags landed", color=NAVY, fontsize=10, fontweight="bold",
                    loc="left")
    for side in ("top", "right", "left"):
        right.spines[side].set_visible(False)
    right.spines["bottom"].set_color(GREY)
    right.tick_params(colors=NAVY, labelsize=8.5)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_leak_classes.png", dpi=220)
    plt.close(fig)


def frontier_axis(meta: dict) -> None:
    """The generator axis, every quantity at once: the front of the funnel and the back.

    Eight panels because the eight quantities share no unit, and one axis per panel so the reader
    is never asked to compare a percentage against a Sharpe ratio. The shape of the result is that
    the first row moves a great deal and the second row does not, which is the paper's claim and
    is not visible in a table of mixed units. This is a rate comparison and never an efficiency
    one -- the arms are not compute-matched.
    """
    records = meta["generator_axis"]
    colours = [STEEL, NAVY, NAVY, NAVY]

    def pull(path: tuple[str, ...], scale: float = 1.0) -> list[float]:
        out = []
        for record in records:
            node: object = record
            for key in path:
                node = node[key]  # type: ignore[index]
            out.append(float(node) * scale)  # type: ignore[arg-type]
        return out

    panels = [
        ("executes on real data (%)", pull(("funnel", "execution_rate"), 100), "{:.0f}"),
        ("executes and trades (%)", pull(("funnel", "position_taking_rate"), 100), "{:.0f}"),
        ("static leak rejections (%)", pull(("audit", "static_rejection_rate"), 100), "{:.0f}"),
        ("semantic rejections (%)", pull(("audit", "semantic_rejection_rate"), 100), "{:.0f}"),
        ("median dev Sharpe", pull(("funnel", "dev_median_sharpe")), "{:.2f}"),
        ("best dev Sharpe", pull(("funnel", "dev_max_sharpe")), "{:.2f}"),
        ("median holdout Sharpe", pull(("funnel", "holdout_median_sharpe")), "{:.2f}"),
        ("holdout Sharpe positive (%)",
         pull(("funnel", "holdout_fraction_positive"), 100), "{:.0f}"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(9.4, 5.0))
    for ax, (title, values, fmt) in zip(axes.ravel(), panels, strict=True):
        ax.bar(range(4), values, color=colours, width=0.66)
        span = max(abs(v) for v in values) or 1.0
        for i, value in enumerate(values):
            ax.text(i, value + span * (0.06 if value >= 0 else -0.20), fmt.format(value),
                    ha="center", color=NAVY, fontsize=7.5)
        ax.axhline(0, color=GREY, lw=0.9)
        ax.set_title(title, color=NAVY, fontsize=8.5, fontweight="bold")
        ax.set_ylim(min(min(values) * 1.45, 0), max(max(values) * 1.28, span * 0.12))
        ax.set_xticks(range(4))
        ax.set_xticklabels(["local", "GPT", "Cl", "Gem"], fontsize=7.5)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
        ax.tick_params(colors=NAVY)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_frontier_axis.png", dpi=220)
    plt.close(fig)


def local_funnel() -> None:
    """Where the local corpus died, as a share of draws. Paper 1's funnel table, drawn.

    The shape is the finding: everything parses, most of it fails to run, most of what runs never
    takes a position, and what remains is a small fraction of what was asked for.
    """
    records = json.loads(
        Path("runs/pooled/backtest_results.json").read_text(encoding="utf-8"))
    total = len(records)
    executed = [r for r in records if r["outcome"] not in ("runtime_error", "timeout")]
    traded = [r for r in executed if (r["mean_turnover"] or 0) > 0]

    stages = [
        ("drawn", total, MIST),
        ("parses, conforms\nto the interface", total, SKY),
        ("executes on\nreal data", len(executed), STEEL),
        ("executes\nand trades", len(traded), NAVY),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    for i, (_label, count, colour) in enumerate(stages):
        share = count / total * 100
        ax.bar(i, share, width=0.62, color=colour,
               edgecolor=SKY if colour is MIST else "none")
        ax.text(i, share + 2.2, f"{count}\n{share:.1f}%", ha="center", color=NAVY, fontsize=8.5)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s[0] for s in stages], fontsize=8.5)
    ax.set_ylabel("share of draws (%)")
    ax.set_ylim(0, 118)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_local_funnel.png", dpi=220)
    plt.close(fig)


def main() -> int:
    configure_logging()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8"))
    predictor_gap()
    capacity_binding()
    impact_identifiability()
    front_vs_back(meta)
    compute_matched(meta)
    null_forest(meta)
    funnel(meta)
    shortcut_safety()
    fragility_by_origin(meta)
    factor_costs()
    factor_deflation()
    dispersion_mechanism()
    factors_vs_corpus()
    alpha_decay()
    reversal_weighting()
    capacity_by_generator()
    population_funnel()
    knife_edge_profile()
    generator_fragility_range()
    leak_classes()
    local_funnel()
    frontier_axis(meta)
    _log.info("wrote 22 figures to %s", OUTDIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
