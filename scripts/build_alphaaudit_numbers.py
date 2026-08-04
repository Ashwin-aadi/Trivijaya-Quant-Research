"""Emit the AlphaAudit paper's generator-validation macros, traced to their artifacts.

AlphaAudit predates the macro pipeline the other two papers use, so its body still states figures
inline. This script does **not** retrofit those; it covers the generator-validation section added
in 2026-08, so that at least the newest figures in that paper obey the rule the other two follow.
The remaining inline figures are a known gap, recorded in the paper's own limitations rather than
quietly left unmentioned.

Macros carry the ``aa`` prefix, matching the namespace convention of the other two papers.

Usage:
    python scripts/build_alphaaudit_numbers.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402
from src.eval.numbers import macro_name  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
POOLED = ROOT / "runs" / "pooled"

Macros = dict[str, tuple[str, str]]

#: Arm key -> the letters used in macro names. LaTeX control sequences may not contain digits, so
#: product names are reduced to bare words and spelled out in the prose instead.
ARM_WORD = {"gpt": "Gpt", "claude": "Claude", "gemini": "Gemini"}


def _put(macros: Macros, name: str, value: str, source: str) -> None:
    """Refuse silent overwrites. Two artifacts claiming one macro is a defect, not a merge."""
    key = macro_name(name)
    if key in macros and macros[key][0] != value:
        raise ValueError(f"macro {key} redefined: {macros[key][0]!r} vs {value!r}")
    macros[key] = (value, source)


def _read(path: Path) -> Any:  # noqa: ANN401 - one artifact is a list, another an object
    return json.loads(path.read_text(encoding="utf-8"))


def _local(macros: Macros) -> None:
    """The local corpus's own audit and holdout figures, the baseline the arms are read against."""
    audit = _read(POOLED / "audit_results.json")
    backtests = _read(POOLED / "backtest_results.json")
    holdout = _read(POOLED / "holdout_results.json")

    rejected = {n for n, v in audit["static"].items() if v["rejected"]}
    total = len(audit["static"])
    _put(macros, "aaLocalDraws", str(total), "runs/pooled/audit_results.json")
    _put(macros, "aaLocalStatic", str(len(rejected)), "runs/pooled/audit_results.json")
    _put(macros, "aaLocalStaticPct", f"{100 * len(rejected) / total:.1f}",
         "runs/pooled/audit_results.json")

    # Rankable is executed *and* traded. A strategy that never took a position has no return
    # series, so it cannot be deflated and cannot be scored out of sample.
    rankable = {
        r["name"] for r in backtests
        if r["outcome"] == "evaluated" and (r.get("mean_turnover") or 0) > 0
    }
    _put(macros, "aaLocalRankable", str(len(rankable)), "runs/pooled/backtest_results.json")
    _put(macros, "aaLocalRankablePct", f"{100 * len(rankable) / total:.1f}",
         "runs/pooled/backtest_results.json")
    hit = len(rankable & rejected)
    _put(macros, "aaLocalStaticRankable", str(hit), "runs/pooled/audit_results.json")
    _put(macros, "aaLocalStaticRankablePct", f"{100 * hit / len(rankable):.1f}",
         "runs/pooled/audit_results.json")

    sharpes = [
        r["sharpe"] for r in holdout
        if r["name"] in rankable and r.get("sharpe") is not None
    ]
    _put(macros, "aaLocalHoldoutSharpe", f"{st.mean(sharpes):.4f}",
         "runs/pooled/holdout_results.json")
    _put(macros, "aaLocalHoldoutN", str(len(sharpes)), "runs/pooled/holdout_results.json")


def _arms(macros: Macros, gv: dict[str, Any]) -> None:
    """Per-arm auditor verdicts and out-of-sample performance."""
    source = "generator_validation.json"
    _put(macros, "aaGvArms", str(gv["n_arms"]), source)
    _put(macros, "aaGvTotal", str(gv["n_total"]), source)
    _put(macros, "aaGvRequestsPerArm", str(gv["requests_per_arm"]), source)
    _put(macros, "aaGvStaticTotal", str(gv["static_total"]), source)
    _put(macros, "aaGvSubsamples", str(gv["n_subsamples"]), source)
    _put(macros, "aaGvBar", f"{gv['dsr_bar']:.2f}", source)
    _put(macros, "aaGvNSmall", str(min(gv["trial_counts"])), source)
    _put(macros, "aaGvNLarge", str(max(gv["trial_counts"])), source)
    _put(macros, "aaGvClearedDev", str(gv["cleared_dev"]), source)
    _put(macros, "aaGvClearedHoldout", str(gv["cleared_holdout"]), source)
    for label, key in (("Small", str(min(gv["trial_counts"]))),
                       ("Large", str(max(gv["trial_counts"])))):
        _put(macros, f"aaGvMatchedDev{label}", str(gv["matched_dev"][key]), source)
        _put(macros, f"aaGvMatchedHoldout{label}", str(gv["matched_holdout"][key]), source)

    for arm, word in ARM_WORD.items():
        row = gv["arms"][arm]
        _put(macros, f"aaGv{word}N", str(row["n"]), source)
        _put(macros, f"aaGv{word}Executed", str(row["executed"]), source)
        _put(macros, f"aaGv{word}Static", str(row["static_rejected"]), source)
        _put(macros, f"aaGv{word}Semantic", str(row["semantic_rejected"]), source)
        _put(macros, f"aaGv{word}Statistical", str(row["statistical_rejected"]), source)
        _put(macros, f"aaGv{word}Pbo", f"{row['pbo']:.4f}", source)
        _put(macros, f"aaGv{word}DevSharpe", f"{row['dev_sharpe_mean']:+.4f}", source)
        _put(macros, f"aaGv{word}HoldSharpe", f"{row['holdout_sharpe_mean']:+.4f}", source)
        _put(macros, f"aaGv{word}HoldBest", f"{row['holdout_sharpe_best']:+.4f}", source)


def _frontier_extra(macros: Macros) -> None:
    """AUAP and the cost burden: P1's primary metric, and what costs did to each arm.

    AUAP appears nowhere in the generator-validation pre-registration. It is computed and reported
    as exploratory, and the paper says so in the same sentence as the figure.
    """
    gaps = _read(ROOT / "data" / "processed" / "frontier_gap_measures.json")
    local_ab = _read(POOLED / "ablation_holdout.json")
    local_costs = _read(POOLED / "gross_vs_net.json")
    csource = "runs/pooled/gross_vs_net.json"

    best = max(local_ab["combinations"], key=lambda c: c["auap"])
    _put(macros, "aaLocalAuapBest", f"{best['auap']:.4f}", "runs/pooled/ablation_holdout.json")
    _put(macros, "aaLocalAuapLayers", " + ".join(best["layers"]),
         "runs/pooled/ablation_holdout.json")
    _put(macros, "aaLocalRandomLow", f"{local_ab['random_baseline_auap_interval'][0]:.4f}",
         "runs/pooled/ablation_holdout.json")
    _put(macros, "aaLocalRandomHigh", f"{local_ab['random_baseline_auap_interval'][1]:.4f}",
         "runs/pooled/ablation_holdout.json")
    _put(macros, "aaLocalCostDrop", f"{local_costs['mean_sharpe_drop']:.4f}", csource)
    _put(macros, "aaLocalFlips", str(local_costs["n_sign_positive_to_negative"]), csource)
    _put(macros, "aaLocalFlipsOf", str(local_costs["n_traded"]), csource)
    _put(macros, "aaLocalFlipsPct",
         f"{100 * local_costs['share_sign_positive_to_negative']:.1f}", csource)

    beats = 0
    cells = 0
    for arm, word in ARM_WORD.items():
        ablation = _read(ROOT / "runs" / f"frontier_{arm}" / "ablation_holdout.json")
        asource = "ablation_holdout.json"
        top = max(ablation["combinations"], key=lambda c: c["auap"])
        _put(macros, f"aaGv{word}AuapBest", f"{top['auap']:.4f}", asource)
        _put(macros, f"aaGv{word}AuapLayers", " + ".join(top["layers"]), asource)
        _put(macros, f"aaGv{word}RandomLow",
             f"{ablation['random_baseline_auap_interval'][0]:.4f}", asource)
        _put(macros, f"aaGv{word}RandomHigh",
             f"{ablation['random_baseline_auap_interval'][1]:.4f}", asource)
        beats += sum(1 for c in ablation["combinations"] if c["beats_random"])
        cells += len(ablation["combinations"])

        costs = gaps["arms"][arm]["gross_vs_net"]
        gsource = "frontier_gap_measures.json"
        _put(macros, f"aaGv{word}CostDrop", f"{costs['mean_sharpe_drop']:.4f}", gsource)
        _put(macros, f"aaGv{word}Flips", str(costs["n_sign_positive_to_negative"]), gsource)
        _put(macros, f"aaGv{word}BestGross", f"{costs['best_gross']:.4f}", gsource)
        _put(macros, f"aaGv{word}BestNet", f"{costs['best_net']:.4f}", gsource)

    _put(macros, "aaGvAuapBeats", str(beats), "ablation_holdout.json")
    _put(macros, "aaGvAuapCells", str(cells), "ablation_holdout.json")
    flips = sum(
        gaps["arms"][a]["gross_vs_net"]["n_sign_positive_to_negative"] for a in ARM_WORD
    )
    _put(macros, "aaGvFlipsTotal", str(flips), "frontier_gap_measures.json")


def main() -> int:
    configure_logging()
    macros: Macros = {}
    _local(macros)
    _arms(macros, _read(ROOT / "data" / "processed" / "generator_validation.json"))
    _frontier_extra(macros)

    lines = [
        "% GENERATED BY scripts/build_alphaaudit_numbers.py -- DO NOT EDIT BY HAND.",
        "% Covers the generator-validation section only. The rest of the AlphaAudit paper",
        "% predates this pipeline and still states its figures inline; that is a known gap.",
        "",
    ]
    for name in sorted(macros):
        value, source = macros[name]
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {source}")
    out = ROOT / "papers" / "alphaaudit_numbers.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bench = ROOT / "benchmarks" / "alphaaudit" / "paper_numbers.json"
    bench.write_text(
        json.dumps({k: {"value": v[0], "source": v[1]} for k, v in sorted(macros.items())},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log.info("wrote %d macros -> %s", len(macros), out.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
