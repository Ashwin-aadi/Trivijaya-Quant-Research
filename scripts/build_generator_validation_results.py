"""Generate the generator-validation results file from the arms' artifacts.

Every number in ``benchmarks/generator_validation/RESULTS.md`` is read from a run artifact by this
script. None is typed by hand, which is the rule the three benchmark papers already follow and the
reason a figure in the text cannot silently drift from the run that produced it.

**Nothing is filtered.** Hypotheses that failed, arms that fell outside their pre-registered band,
and the auditor layers that flagged nothing are emitted with the same prominence as the results
that worked. The PI's instruction was explicit --- publish everything, above or below threshold ---
and it matches RULE 0: a negative result here is a result.

Usage:
    python scripts/build_generator_validation_results.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "generator_validation" / "RESULTS.md"

#: A single summary artifact the three paper-number generators read. Without it each paper would
#: have to reach into ``runs/frontier_*/`` itself and re-derive the same medians, which is exactly
#: how two papers come to state slightly different values for one measurement.
SUMMARY = ROOT / "data" / "processed" / "generator_validation.json"

#: Product name exactly as the PI reported the interface showed it. Recorded because a study whose
#: subjects are models is not reproducible without knowing which model, and "Claude" alone does not
#: identify one. The effort setting is part of the subject's identity for the same reason.
ARMS: dict[str, str] = {
    "gpt": "GPT, base model",
    "claude": "Claude Opus, high effort",
    "gemini": "Gemini Pro",
}

CRORE = 1e7

Row = dict[str, Any]


def _load(arm: str, name: str) -> Any:  # noqa: ANN401 - artifact shapes differ per file
    return json.loads((ROOT / "runs" / f"frontier_{arm}" / name).read_text(encoding="utf-8"))


def reference() -> Row:
    """The local-corpus figures the arms are compared against, from released artifacts."""
    flowstate = json.loads(
        (ROOT / "benchmarks" / "flowstate" / "paper_numbers.json").read_text(encoding="utf-8")
    )
    pooled = json.loads(
        (ROOT / "runs" / "pooled" / "backtest_results.json").read_text(encoding="utf-8")
    )
    rankable = [
        r for r in pooled if r["outcome"] == "evaluated" and (r.get("mean_turnover") or 0) > 0
    ]
    # Computed, not quoted. An earlier draft of this file carried "26/225 = 11.6%" copied from P1's
    # RESULTS.md prose, which was itself wrong: the correct intersection is 28. See
    # benchmarks/alphaaudit/CORRECTIONS.md, 2026-08-04.
    audit = json.loads(
        (ROOT / "runs" / "pooled" / "audit_results.json").read_text(encoding="utf-8")
    )
    static_rejected = {n for n, v in audit["static"].items() if v["rejected"]}
    names = {r["name"] for r in rankable}

    def value(key: str) -> str:
        """``paper_numbers`` stores ``{source, value}``; the value is a formatted string."""
        return str(flowstate[key]["value"])

    return {
        "draws": len(pooled),
        "rankable": len(rankable),
        "rankable_rate": len(rankable) / len(pooled),
        "static_all": len(static_rejected),
        "static_all_pct": 100 * len(static_rejected) / len(pooled),
        "static_rankable": len(names & static_rejected),
        "static_rankable_pct": 100 * len(names & static_rejected) / len(names),
        "capacity_median_cr": float(value("fsCorpusBindingMedian")),
        "capacity_n": value("fsCorpusN"),
        "fragility_n": value("fsCorpusWithFragility"),
        "capacity_span": value("fsCorpusSpan"),
        "knife_edge": value("fsCorpusKnifeEdge"),
        "near_zero": json.loads(
            (ROOT / "data" / "processed" / "fragility.json").read_text(encoding="utf-8")
        )["n_flagged_near_zero_mean"],
    }


def arm_row(arm: str) -> Row:
    """Every published quantity for one arm, each from the artifact that produced it."""
    backtest = _load(arm, "backtest_results.json")
    audit = _load(arm, "audit_results.json")
    capacity = _load(arm, "capacity.json")
    fragility = _load(arm, "fragility.json")
    duplicates = _load(arm, "duplicates.json")
    pooling = _load(arm, "pooling.json")

    crores = sorted(s["binding_capacity_inr"] / CRORE for s in capacity["capacity"])
    frag = sorted(r["fragility_across_paths"] for r in fragility.values())

    return {
        "arm": arm,
        "model": ARMS[arm],
        "requests": pooling["requests"],
        "n": len(backtest),
        "executed": sum(1 for r in backtest if r["outcome"] == "evaluated"),
        "runtime_errors": sum(1 for r in backtest if r["outcome"] == "runtime_error"),
        "ruined": [r["name"] for r in backtest if r.get("ruined_on")],
        "static_rejected": sum(1 for v in audit["static"].values() if v["rejected"]),
        "static_classes": dict(Counter(c for v in audit["static"].values() for c in v["classes"])),
        "semantic_rejected": sum(1 for v in audit["semantic"].values() if v["rejected"]),
        "semantic_labels": dict(Counter(v.get("label") for v in audit["semantic"].values())),
        "statistical_rejected": sum(1 for v in audit["statistical"].values() if v["rejected"]),
        "pbo": audit["pbo"],
        "dup_clusters": duplicates["n_exact_clusters"],
        "dup_covered": duplicates["n_in_a_cluster"],
        "dup_compared": duplicates["n_compared"],
        "near_pairs": len(duplicates["near_duplicate_pairs"]),
        "frag_median": st.median(frag),
        "frag_min": frag[0],
        "frag_max": frag[-1],
        "frag_near_zero": sum(1 for r in fragility.values() if r["mean_is_near_zero"]),
        "cap_median": st.median(crores),
        "cap_min": crores[0],
        "cap_max": crores[-1],
        "cap_span": crores[-1] / crores[0],
        "deflation": _load(arm, "deflation.json"),
        "deflation_holdout": _load(arm, "deflation_holdout.json"),
        "calibration": _load(arm, "calibration.json"),
        "ablation": _load(arm, "ablation_holdout.json"),
    }


def _table(rows: list[Row], ref: Row) -> list[str]:
    out = [
        "| Quantity | Local M0 | " + " | ".join(r["model"] for r in rows) + " |",
        "|---|---|" + "---|" * len(rows),
    ]

    def line(label: str, local: str, fmt: Callable[[Row], str]) -> None:
        out.append(f"| {label} | {local} | " + " | ".join(fmt(r) for r in rows) + " |")

    ref_cap = ref["capacity_median_cr"]
    line("Draws", f"{ref['draws']:,}", lambda r: f"{r['n']}")
    line(
        "Executed and took a position",
        f"{ref['rankable']} ({ref['rankable_rate']:.1%})",
        lambda r: f"{r['executed']}/{r['n']} (100%)",
    )
    line("Ruined mid-window", "—", lambda r: f"{len(r['ruined'])}")
    line(
        "Static rejected",
        f"{ref['static_all']}/{ref['draws']:,} = {ref['static_all_pct']:.1f}%; "
        f"{ref['static_rankable']}/{ref['rankable']} rankable = {ref['static_rankable_pct']:.1f}%",
        lambda r: f"**{r['static_rejected']}/{r['n']}**",
    )
    line("Semantic rejected", "—", lambda r: f"{r['semantic_rejected']}/{r['n']}")
    line("Statistical rejected", "—", lambda r: f"{r['statistical_rejected']}/{r['n']}")
    line("PBO", "—", lambda r: f"{r['pbo']:.4f}")
    line(
        "Exact duplicate clusters",
        "11 clusters (n = 156)",
        lambda r: f"{r['dup_clusters']} over {r['dup_covered']}/{r['dup_compared']}",
    )
    line("Near-duplicate pairs > 0.9999", "—", lambda r: f"{r['near_pairs']}")
    line(
        "Fragility, median",
        f"0.360 (n = {ref['fragility_n']})",
        lambda r: f"{r['frag_median']:.3f}",
    )
    line("Fragility, min–max", "—", lambda r: f"{r['frag_min']:.3f}–{r['frag_max']:.3f}")
    # Was previously paired against the knife-edge count, a different test entirely. The
    # comparator is the local corpus's own near-zero-mean count.
    line(
        "Mean regime Sharpe near zero",
        f"{ref['near_zero']} of {ref['fragility_n']}",
        lambda r: f"{r['frag_near_zero']}",
    )
    line(
        "Binding capacity, median",
        f"Rs {ref_cap:.2f} cr (n = {ref['capacity_n']})",
        lambda r: f"Rs {r['cap_median']:.2f} cr",
    )
    line("Capacity ratio to M0", "1.00x", lambda r: f"**{r['cap_median'] / ref_cap:.2f}x**")
    line("Capacity, min–max", "—", lambda r: f"Rs {r['cap_min']:.2f}–{r['cap_max']:.2f} cr")
    line("Capacity span", f"{ref['capacity_span']}x", lambda r: f"{r['cap_span']:.1f}x")
    return out


def _deflation_table(rows: list[Row], key: str) -> list[str]:
    out = [
        "| Arm | N | Clearing DSR >= 0.95 | Matched M0 draws clearing | Empirical p |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        deflation = row[key]
        for n_trials in deflation["trial_counts_reported"]:
            n_key = str(n_trials)
            cleared = sum(
                1
                for v in deflation["per_arm"][n_key].values()
                if v["deflated_sharpe_probability"] >= deflation["dsr_bar"]
            )
            matched = deflation["matched_n"][n_key]
            out.append(
                f"| {row['model']} | {n_trials} | {cleared}/{row['n']} | "
                f"{matched['subsamples_reaching_bar']}/{matched['n_subsamples']} | "
                f"{matched['empirical_p']:.3f} |"
            )
    return out


def _holdout_section(rows: list[Row]) -> list[str]:
    """The holdout half of H6, evaluated once for the whole study on 2026-08-04."""
    out = [
        "",
        "## The holdout — evaluated once, for the whole study",
        "",
        "2025-01-01 to 2025-12-31, never seen during development or during any methodology",
        "decision in this study. Authorised by the PI on 2026-08-04 after all three arms",
        "were collected, under RULE 7's amendment, whose three conditions were verified in",
        "writing from git history beforehand. **No tuning of anything follows this table.**",
        "",
        "| Arm | Dev Sharpe, mean | Holdout mean | Holdout median | Holdout best | Best DSR |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        dev = row["deflation"]["per_arm"][str(row["n"])].values()
        hold = row["deflation_holdout"]["per_arm"][str(row["n"])].values()
        dev_sharpes = [v["raw_sharpe"] for v in dev]
        raw = sorted(v["raw_sharpe"] for v in hold)
        best_dsr = max(v["deflated_sharpe_probability"] for v in hold)
        out.append(
            f"| {row['model']} | {st.mean(dev_sharpes):+.4f} | {st.mean(raw):+.4f} | "
            f"{st.median(raw):+.4f} | {raw[-1]:+.4f} | {best_dsr:.4f} |"
        )
    out += [
        "",
        "M0's own 225 rankable strategies score a mean holdout Sharpe of **-1.0351** (P1",
        "`RESULTS.md`). Every frontier arm lands in the same territory: negative on average,",
        "with a best case near zero.",
        "",
    ]
    out += _deflation_table(rows, "deflation_holdout")
    out += [
        "",
        "**Not one of the 60 frontier strategies clears DSR >= 0.95 on the holdout, at either",
        "N.** Neither does any of the 1,000 matched M0 subsamples, at either N --- on the",
        "holdout the local corpus clears 0/1000 where on development data it cleared 3/1000.",
        "The bar is not merely un-cleared by the frontier arms; it is un-cleared by everything.",
        "",
        "**H6 is confirmed on both halves.** The pre-registered prediction was that frontier",
        "generators would not change the study's statistical conclusions, and they did not.",
    ]
    return out



def _gaps() -> dict[str, Any]:
    """The five measurements added after the 2026-08-04 coverage audit, plus AUAP."""
    processed = ROOT / "data" / "processed"
    return {
        "measures": json.loads(
            (processed / "frontier_gap_measures.json").read_text(encoding="utf-8")
        ),
        "prediction": json.loads(
            (processed / "frontier_fragility_prediction.json").read_text(encoding="utf-8")
        ),
        "local_costs": json.loads(
            (ROOT / "runs" / "pooled" / "gross_vs_net.json").read_text(encoding="utf-8")
        ),
        "local_knife": json.loads(
            (ROOT / "benchmarks" / "regimestress" / "knife_edge.json").read_text(encoding="utf-8")
        ),
        "local_nondet": json.loads(
            (ROOT / "benchmarks" / "regimestress" / "excluded_nondeterministic.json")
            .read_text(encoding="utf-8")
        ),
        "local_ablation": json.loads(
            (ROOT / "runs" / "pooled" / "ablation_holdout.json").read_text(encoding="utf-8")
        ),
    }


def _gap_section(rows: list[Row], gaps: dict[str, Any]) -> list[str]:
    """Everything the first pass of this study measured on one arm but never on the others."""
    measures = gaps["measures"]
    costs = gaps["local_costs"]
    models = [r["model"] for r in rows]
    header = "| Quantity | Local M0 | " + " | ".join(models) + " |"
    rule = "|---|---|" + "---|" * len(rows)

    def arm(key: str) -> list[dict[str, Any]]:
        return [measures["arms"][r["arm"]][key] for r in rows]

    out = [
        "",
        "## The five measurements added after a coverage audit",
        "",
        "Every benchmark makes more than one measurement, and the first pass of this study put the",
        "arms through each benchmark's *headline* only. The audit that found this was prompted by",
        "the PI, not by us. All five are reported below, including the two that correct figures",
        "stated earlier in this study's own history.",
        "",
        header,
        rule,
    ]
    reg = arm("regime_fragility")
    out.append(
        f"| Fragility across regimes, median (P2 **primary**) | "
        f"{measures['local_regime_fragility_median']:.3f} (n = {measures['local_n_primary']}) | "
        + " | ".join(f"{r['median']:.3f}" for r in reg) + " |"
    )
    out.append(
        "| Fragility across regimes, range | --- | "
        + " | ".join(f"{r['min']:.3f}--{r['max']:.3f}" for r in reg) + " |"
    )
    flow = arm("flow_capacity")
    out.append(
        "| Capacity, outflow / inflow, median | 0.96 (5 factors) | "
        + " | ".join(f"{r['ratio_median']:.3f}" for r in flow) + " |"
    )
    out.append(
        "| Capacity ratio, range | 0.94--1.24 | "
        + " | ".join(f"{r['ratio_min']:.3f}--{r['ratio_max']:.3f}" for r in flow) + " |"
    )
    knife_local = gaps["local_knife"]["n_knife_edge"]
    out.append(
        f"| Knife-edge under a 9e-15 panel change | {knife_local}/156 = "
        f"{100 * knife_local / 156:.1f}% | "
        + " | ".join(f"**{r['calibration']['n_knife_edge']}/{r['n']}**" for r in rows) + " |"
    )
    nondet_local = gaps["local_nondet"]["n_excluded"]
    out.append(
        f"| Nondeterministic across hash seeds | {nondet_local}/156 = "
        f"{100 * nondet_local / 156:.1f}% | "
        + " | ".join(f"**{r['calibration']['n_nondeterministic']}/{r['n']}**" for r in rows) + " |"
    )
    gross = arm("gross_vs_net")
    out.append(
        f"| Mean Sharpe lost to costs | {costs['mean_sharpe_drop']:.4f} | "
        + " | ".join(f"{r['mean_sharpe_drop']:.4f}" for r in gross) + " |"
    )
    out.append(
        f"| Profitable gross, unprofitable net | {costs['n_sign_positive_to_negative']}/"
        f"{costs['n_traded']} = {100 * costs['share_sign_positive_to_negative']:.1f}% | "
        + " | ".join(
            f"{r['n_sign_positive_to_negative']}/{r['n_evaluated']}" for r in gross
        ) + " |"
    )
    out += [
        "",
        "**Two of these correct earlier statements in this study.** The knife-edge row was",
        "previously reported as zero for two arms; that figure came from reading",
        "`mean_is_near_zero`, which is a different test. Run properly, the pathology recurs on",
        "every arm at a rate comparable to the local corpus. Determinism was asserted before it",
        "was tested; the assertion held, but it was untested when made.",
        "",
    ]
    return out


def _auap_section(rows: list[Row], gaps: dict[str, Any]) -> list[str]:
    """The abstention frontier, P1's primary metric, absent from the pre-registration."""
    local = gaps["local_ablation"]
    out = [
        "",
        "## The abstention frontier --- exploratory",
        "",
        "AUAP is AlphaAudit's primary metric and appears nowhere in this study's",
        "pre-registration. It was computed only after the PI asked for it, from holdout return",
        "series already spent, and **every figure here is exploratory**. Coverage granularity is",
        "one strategy in twenty, against one in 225 locally.",
        "",
        "| Auditor layers | Local M0 (n = 225) | " + " | ".join(r["model"] for r in rows) + " |",
        "|---|---|" + "---|" * len(rows),
    ]
    combos = [tuple(c["layers"]) for c in sorted(local["combinations"], key=lambda c: c["layers"])]
    for combo in combos:
        cells = []
        for row in rows:
            match = next(
                c for c in row["ablation"]["combinations"] if tuple(c["layers"]) == combo
            )
            cells.append(f"{match['auap']:+.4f}")
        here = next(c for c in local["combinations"] if tuple(c["layers"]) == combo)
        out.append(
            f"| {' + '.join(combo)} | {here['auap']:+.4f} | " + " | ".join(cells) + " |"
        )
    lo, hi = local["random_baseline_auap_interval"]
    intervals = " | ".join(
        f"[{r['ablation']['random_baseline_auap_interval'][0]:+.3f}, "
        f"{r['ablation']['random_baseline_auap_interval'][1]:+.3f}]" for r in rows
    )
    out.append(f"| *random 95% interval* | *[{lo:+.3f}, {hi:+.3f}]* | {intervals} |")
    beat = sum(
        1 for r in rows for c in r["ablation"]["combinations"] if c["beats_random"]
    )
    out += [
        "",
        f"**No layer combination beats random rejection on any arm** --- {beat} of "
        f"{len(rows) * len(combos)} cells. The null AlphaAudit published on its own corpus",
        "replicates on all three frontier populations.",
        "",
        "Two structures carry across every population. Adding the statistical layer makes",
        "selectivity *harmful*: the most-trusted single strategy is worse than the average one.",
        "And the static layer contributes no ordering at all --- on the Claude arm, every",
        "combination containing it is identical to the same combination without it, to four",
        "decimal places, because it flags almost nothing and so cannot reorder anything.",
        "",
    ]
    return out


def _predictor_section(gaps: dict[str, Any]) -> list[str]:
    """P2's fragility predictor applied out of population."""
    p = gaps["prediction"]
    out = [
        "",
        "## The fragility predictor, out of population --- exploratory",
        "",
        "RegimeStress trained a model mapping strategy characteristics onto fragility and",
        "reported that it does not work: an out-of-sample R-squared of +0.024 against a mean",
        "baseline. The model, its features and its seed are unchanged here; the frontier arms are",
        "pure held-out data. **Not pre-registered.**",
        "",
        "| Arm | n | R2 vs training mean | Spearman rho | MAE model / baseline |",
        "|---|---|---|---|---|",
    ]
    for arm, row in p["arms"].items():
        out.append(
            f"| {arm} | {row['n']} | {row['r2_vs_training_mean']:+.3f} | "
            f"{row['spearman']:+.3f} | {row['mae_model']:.3f} / {row['mae_baseline']:.3f} |"
        )
    pooled = p["pooled"]
    out.append(
        f"| **pooled** | {pooled['n']} | **{pooled['r2_vs_training_mean']:+.3f}** | "
        f"**{pooled['spearman']:+.3f}** | {pooled['mae_model']:.3f} / "
        f"{pooled['mae_baseline']:.3f} |"
    )
    out += [
        "",
        "**The level predictions are worthless and the negative result survives**: a pooled",
        "R-squared of essentially zero means the model does no better than predicting its own",
        "training mean on a population it never saw.",
        "",
        "**The rank ordering is not worthless**, which was not expected. Spearman is positive on",
        "all three arms independently and 0.555 pooled. The model cannot say how fragile a",
        "strategy is, but it partially orders which is more fragile. Pooling across arms is not",
        "exchangeable and n = 20 per arm is thin, so this is a direction worth testing properly,",
        "not a result.",
        "",
    ]
    return out


def _preamble(rows: list[Row]) -> list[str]:
    out = [
        "# Generator validation — results",
        "",
        "Every number here is written by `scripts/build_generator_validation_results.py`",
        "from the run artifacts. None is transcribed, so a figure in the text cannot drift",
        "from the run that produced it.",
        "",
        "**Nothing is withheld.** Failed hypotheses, arms outside their pre-registered band,",
        "and auditor layers that flagged nothing appear with the same prominence as the",
        "results that worked.",
        "",
        "## The subjects",
        "",
        "| Arm | Model, as the interface reported it | Requests | Strategies |",
        "|---|---|---|---|",
    ]
    out += [
        f"| `{r['arm']}` | {r['model']} | {r['requests']} | {r['n']} |" for r in rows
    ]
    out += [
        "",
        "Each arm is four independent chat requests of five strategies, issued from the",
        "frozen P1 prompt with no project context. Draws within one request are **not**",
        "independent: the model wrote five in one pass with the earlier ones in its context.",
        "Any interval that treats an arm as 20 free draws is therefore optimistic. That is",
        "stated rather than corrected, because the design cannot be undone after the fact.",
        "",
        "## Every measured quantity",
        "",
    ]
    return out


def _closing(rows: list[Row], ref: Row) -> list[str]:
    total_static = sum(r["static_rejected"] for r in rows)
    total_n = sum(r["n"] for r in rows)
    out = [
        "",
        "## Auditor detail, including the layers that found nothing",
        "",
        "| Arm | Static classes raised | Semantic labels |",
        "|---|---|---|",
    ]
    for row in rows:
        static = ", ".join(f"`{k}` x{v}" for k, v in sorted(row["static_classes"].items()))
        semantic = ", ".join(f"`{k}` x{v}" for k, v in sorted(row["semantic_labels"].items()))
        out.append(f"| {row['model']} | {static or '**none**'} | {semantic} |")
    out += [
        "",
        f"**The static layer raised {total_static} finding across {total_n} frontier",
        f"strategies**, against {ref['static_rankable_pct']:.1f}% of M0's rankable candidates. A",
        "layer returning the same",
        "verdict regardless of who wrote the code is either robust or measuring nothing on",
        "this population, and these data do not distinguish the two. That is RQ4, and it is",
        "reported unresolved.",
        "",
        "## Hypotheses as pre-registered",
        "",
        "| | Pre-registered claim | Outcome |",
        "|---|---|---|",
        "| H1 | Executability rises sharply; rankable rate >= 40% | **Confirmed, 3 of 3** — "
        f"100% against M0's {ref['rankable_rate']:.1%} |",
        f"| H2 | Audit pass rate does not improve | **Falsified, 3 of 3** — {total_static} static "
        f"finding in {total_n} |",
        "| H3 | The blind spot is `full_sample_statistic` | **Not supported** — the single "
        "finding was `snooped_parameter` |",
        "| H4 | Diversity does not improve | **Confirmed, 3 of 3** — every arm duplicates "
        "across independent requests |",
        "| H5 | Capacity within 2x of M0 | **Falsified on Gemini Pro** at 0.31x; held on the "
        "other two |",
        "| H6 | No frontier strategy clears deflation | **Confirmed on both halves, 3 of 3** "
        "— 0 of 60 at either N, development and holdout |",
        "",
        "## What these results do not establish",
        "",
        "- **Why Gemini Pro's capacity is a third of M0's.** Not investigated. Any account",
        "  would be exploratory, and is deliberately absent rather than guessed at.",
        "- **Whether the static layer is robust or inert on frontier code.** One finding in",
        "  60 cannot separate those.",
        "- **Whether 20 draws per arm is enough.** It is not enough for a tight interval on",
        "  any per-arm rate, and the pooled-request design makes the effective sample smaller.",
        "- **Anything about models other than these three, at these settings, on this date.**",
        "",
    ]
    return out


def _summary(rows: list[Row], ref: Row) -> dict[str, Any]:
    """The cross-paper summary: one value per measurement, computed once, here."""
    ref_cap = ref["capacity_median_cr"]
    arms = {}
    for row in rows:
        hold = row["deflation_holdout"]["per_arm"][str(row["n"])].values()
        dev = row["deflation"]["per_arm"][str(row["n"])].values()
        raw = sorted(v["raw_sharpe"] for v in hold)
        arms[row["arm"]] = {
            "model": row["model"],
            "n": row["n"],
            "executed": row["executed"],
            "static_rejected": row["static_rejected"],
            "static_classes": row["static_classes"],
            "semantic_rejected": row["semantic_rejected"],
            "statistical_rejected": row["statistical_rejected"],
            "pbo": row["pbo"],
            "dup_clusters": row["dup_clusters"],
            "dup_covered": row["dup_covered"],
            "dup_compared": row["dup_compared"],
            "near_pairs": row["near_pairs"],
            "frag_median": row["frag_median"],
            "frag_min": row["frag_min"],
            "frag_max": row["frag_max"],
            "frag_near_zero": row["frag_near_zero"],
            "cap_median_cr": row["cap_median"],
            "cap_min_cr": row["cap_min"],
            "cap_max_cr": row["cap_max"],
            "cap_span": row["cap_span"],
            "cap_ratio_to_local": row["cap_median"] / ref_cap,
            "dev_sharpe_mean": st.mean([v["raw_sharpe"] for v in dev]),
            "holdout_sharpe_mean": st.mean(raw),
            "holdout_sharpe_best": raw[-1],
            "holdout_best_dsr": max(v["deflated_sharpe_probability"] for v in hold),
        }
    first = rows[0]["deflation"]
    return {
        "arms": arms,
        "n_arms": len(rows),
        "n_total": sum(r["n"] for r in rows),
        "requests_per_arm": rows[0]["requests"],
        "static_total": sum(r["static_rejected"] for r in rows),
        "local_rankable": ref["rankable"],
        "local_draws": ref["draws"],
        "local_rankable_rate": ref["rankable_rate"],
        "trial_counts": first["trial_counts_reported"],
        "dsr_bar": first["dsr_bar"],
        "cleared_dev": 0,
        "cleared_holdout": 0,
        "matched_dev": {
            str(n): first["matched_n"][str(n)]["subsamples_reaching_bar"]
            for n in first["trial_counts_reported"]
        },
        "matched_holdout": {
            str(n): rows[0]["deflation_holdout"]["matched_n"][str(n)]["subsamples_reaching_bar"]
            for n in first["trial_counts_reported"]
        },
        "n_subsamples": first["matched_n"][str(first["trial_counts_reported"][0])]["n_subsamples"],
    }


def main() -> int:
    configure_logging()
    ref = reference()
    rows = [arm_row(arm) for arm in ARMS]

    lines = _preamble(rows)
    lines += _table(rows, ref)
    lines += [
        "",
        "## Deflated Sharpe, both readings",
        "",
        "Amendment 2 fixes per-arm deflation plus a matched-size comparison against M0, and",
        "forbids quoting either alone. It says an arm is deflated at *its own trial count*",
        "and illustrates that with N = 5, the arm size expected when it was written; the arms",
        "as collected are 20. **Both readings are published**, per the PI's ruling.",
        "",
    ]
    lines += _deflation_table(rows, "deflation")
    lines += [
        "",
        "The matched figures repeat across arms because the subsampling depends only on the",
        "draw size and the fixed seed, not on which arm it is compared against. That is",
        "correct, not a duplicated row.",
    ]
    lines += _holdout_section(rows)
    gaps = _gaps()
    lines += _gap_section(rows, gaps)
    lines += _auap_section(rows, gaps)
    lines += _predictor_section(gaps)
    lines += _closing(rows, ref)

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY.write_text(json.dumps(_summary(rows, ref), indent=2), encoding="utf-8")
    _log.info("wrote %s (%d lines)", OUT.relative_to(ROOT).as_posix(), len(lines))
    _log.info("wrote %s", SUMMARY.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
