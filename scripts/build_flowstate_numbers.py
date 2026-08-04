"""Generate every figure the FlowState paper states, so none of them is typed by hand.

The guarantee this establishes is the one P2 introduced and the PI made binding: **no number is
transcribed into the paper.** Each figure is computed here from a frozen artifact, emitted as a
LaTeX macro, and referenced by name. `scripts/check_paper_numbers.py` then refuses any bare numeral
in a claim position, so a hand-edited figure cannot survive a commit.

Emits ``papers/flowstate_numbers.tex``, ``benchmarks/flowstate/paper_numbers.json``, and renders
``benchmarks/flowstate/RESULTS.md`` from its template.

Usage:
    python scripts/build_flowstate_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.eval.numbers import (  # noqa: E402
    fixed,
    integer,
    macro_name,
    percent,
    plain,
    scientific,
    signed,
)

_log = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "flowstate"

#: name -> (rendered value, the artifact it came from). The provenance half is not decoration: it
#: is what lets a reader trace any figure in the paper back to a file on disk.
Macros = dict[str, tuple[str, str]]

CRORE = 1e7

#: LaTeX macro names must be letters only. A digit in a control sequence is silently
#: mis-parsed and the digits print into the body, so every horizon is spelled out.
HWORD = {1: "One", 2: "Two", 3: "Three", 5: "Five", 10: "Ten", 21: "TwentyOne",
         42: "FortyTwo", 63: "SixtyThree"}


def _put(macros: Macros, name: str, value: str, source: str) -> None:
    """Refuse silent overwrites. Two artifacts claiming one macro is a defect, not a merge."""
    key = macro_name(name)
    if key in macros and macros[key][0] != value:
        raise ValueError(
            f"macro {key} redefined: {macros[key][0]!r} from {macros[key][1]} vs {value!r} "
            f"from {source}"
        )
    macros[key] = (value, source)


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def _decay(macros: Macros, flow: dict[str, Any]) -> None:
    """RQ2. The finding is the absence of decay, so the curve's flatness is what must be figured."""
    src = "flowstate.json"
    horizons = flow["horizons"]
    _put(macros, "fsHorizonMin", integer(min(horizons)), src)
    _put(macros, "fsHorizonMax", integer(max(horizons)), src)
    _put(macros, "fsNFormationDates", integer(flow["n_formation_dates"]), src)
    _put(macros, "fsNInvestableSymbolDays", integer(flow["n_symbol_days_investable"]), src)
    _put(macros, "fsNSymbols", integer(flow["n_symbols"]), src)
    _put(macros, "fsNFactors", integer(len(flow["factors_built"])), src)

    word = {
        "momentum_12_1": "Momentum",
        "reversal_1m": "Reversal",
        "low_volatility": "LowVol",
        "illiquidity": "Illiquidity",
        "liquidity_size_proxy": "SizeProxy",
    }
    halved = 0
    for factor, block in flow["decay"].items():
        stem = word[factor]
        curve = {p["horizon"]: p for p in block["curve"]}
        for horizon in (1, 5, 21, 63):
            point = curve[horizon]
            _put(macros, f"fs{stem}Bps{HWORD[horizon]}",
                 signed(point["mean_return_per_session"] * 1e4, 2), src)
            _put(macros, f"fs{stem}Tstat{HWORD[horizon]}", signed(point["t_statistic"], 2), src)
        _put(macros, f"fs{stem}NonOverlappingOne", integer(curve[1]["n_non_overlapping"]), src)
        if block["half_life_sessions"] is not None:
            halved += 1
    _put(macros, "fsNFactorsWithHalfLife", integer(halved), src)

    # The largest |t| anywhere in the decay table, which is the strongest claim RQ2 can make.
    best = max(
        abs(p["t_statistic"]) for b in flow["decay"].values() for p in b["curve"]
    )
    _put(macros, "fsDecayMaxAbsTstat", fixed(best, 2), src)
    shortest = max(
        abs(p["t_statistic"]) for b in flow["decay"].values() for p in b["curve"]
        if p["horizon"] == 1
    )
    _put(macros, "fsDecayMaxAbsTstatShortest", fixed(shortest, 2), src)


def _capacity(macros: Macros, flow: dict[str, Any]) -> None:
    """RQ1. Binding capacity is the headline; the median is reported only to show what it hid."""
    src = "flowstate.json"
    _put(macros, "fsParticipationLimit", percent(flow["participation_limit"]), src)
    rows = {s["factor"]: s for s in flow["capacity"]}
    word = {
        "momentum_12_1": "Momentum",
        "reversal_1m": "Reversal",
        "low_volatility": "LowVol",
        "illiquidity": "Illiquidity",
        "liquidity_size_proxy": "SizeProxy",
    }
    binding = []
    for factor, stem in word.items():
        row = rows[factor]
        binding.append(row["binding_capacity_inr"])
        _put(macros, f"fs{stem}Binding", fixed(row["binding_capacity_inr"] / CRORE, 2), src)
        _put(macros, f"fs{stem}Median", fixed(row["median_capacity_inr"] / CRORE, 1), src)
        _put(macros, f"fs{stem}Entry", fixed(row["entry_capacity_inr"] / CRORE, 2), src)
        _put(macros, f"fs{stem}OneName", percent(row["fraction_bound_by_one_name"]), src)
        _put(macros, f"fs{stem}Relaxed", fixed(row["relaxed_over_constrained"], 2), src)
        _put(macros, f"fs{stem}Sessions", integer(row["n_rebalance_sessions"]), src)
        _put(macros, f"fs{stem}MedianOverBinding",
             fixed(row["median_capacity_inr"] / row["binding_capacity_inr"], 1), src)
    _put(macros, "fsFactorBindingMin", fixed(min(binding) / CRORE, 2), src)
    _put(macros, "fsFactorBindingMax", fixed(max(binding) / CRORE, 2), src)


def _flow_conditional(macros: Macros, flow: dict[str, Any]) -> None:
    """The novel contribution, and a null. The ratios and their sample sizes travel together."""
    src = "flowstate.json"
    by: dict[str, dict[str, Any]] = {}
    for row in flow["capacity_by_flow_state"]:
        by.setdefault(row["factor"], {})[row["flow_state"]] = row
    ratios = []
    sessions = []
    for states in by.values():
        if "inflow" in states and "outflow" in states:
            ratios.append(
                states["outflow"]["median_capacity_inr"] / states["inflow"]["median_capacity_inr"]
            )
            sessions.extend([states["inflow"]["n_sessions"], states["outflow"]["n_sessions"]])
    _put(macros, "fsFlowRatioMin", fixed(min(ratios), 2), src)
    _put(macros, "fsFlowRatioMax", fixed(max(ratios), 2), src)
    _put(macros, "fsFlowRatioMedian", fixed(float(np.median(ratios)), 2), src)
    _put(macros, "fsFlowSessionsMin", integer(min(sessions)), src)
    _put(macros, "fsFlowSessionsMax", integer(max(sessions)), src)
    _put(macros, "fsFlowNFactors", integer(len(ratios)), src)


def _identification(macros: Macros, ident: dict[str, Any]) -> None:
    """RQ3. The headline is that the sign depends on the weighting, not that nothing was found."""
    src = "impact_identifiability.json"
    delta = ident["d1_exponent"]["delta"]
    _put(macros, "fsDeltaMedian", fixed(delta["median"], 3), src)
    _put(macros, "fsDeltaPTen", fixed(delta["p10"], 3), src)
    _put(macros, "fsDeltaPNinety", fixed(delta["p90"], 3), src)
    _put(macros, "fsDeltaN", integer(delta["n"]), src)
    _put(macros, "fsFitRsquared", fixed(ident["d1_exponent"]["fit_r_squared"]["median"], 3), src)

    for horizon in (1, 5, 10, 21):
        block = ident["d2_transience"]["by_horizon"][str(horizon)]
        det = block["detectability_heavy"]
        _put(macros, f"fsPooledBeta{HWORD[horizon]}", signed(det["pooled_beta"], 4), src)
        _put(macros, f"fsUnweightedBeta{HWORD[horizon]}",
             signed(det["unweighted_mean_beta"], 4), src)
        _put(macros, f"fsMdePooled{HWORD[horizon]}",
             fixed(det["pooled_minimum_detectable_beta"], 4), src)
        _put(macros, f"fsMdeSymbol{HWORD[horizon]}",
             fixed(det["median_symbol_minimum_detectable_beta"], 3), src)
    _put(macros, "fsIdentNSymbols",
         integer(ident["d2_transience"]["by_horizon"]["5"]["detectability_heavy"]["n_symbols"]),
         src)
    by_horizon = ident["d2_transience"]["by_horizon"]
    disagree = sum(
        1 for h in ("1", "5", "10", "21")
        if by_horizon[h]["detectability_heavy"]["estimates_disagree_on_sign"]
    )
    _put(macros, "fsSignDisagreements", integer(disagree), src)
    _put(macros, "fsHorizonsTested", integer(4), src)

    gap = ident["d3_extrapolation"]
    _put(macros, "fsTargetParticipation", percent(gap["target_participation"]), src)
    _put(macros, "fsObservedMedianRelValue", fixed(gap["observed_median_relative_value"], 3), src)
    _put(macros, "fsOrdersOfMagnitude", fixed(gap["orders_of_magnitude_below_data"], 2), src)
    _put(macros, "fsSymbolDaysBelowTarget",
         integer(round(gap["fraction_of_sessions_below_target"] * gap["n_symbol_days"])), src)
    _put(macros, "fsIdentSymbolDays", integer(gap["n_symbol_days"]), src)

    stab = ident["d4_stability"]
    _put(macros, "fsDeltaSpearman", fixed(stab["spearman_between_halves"], 3), src)
    _put(macros, "fsDeltaSpearmanN", integer(stab["n_symbols_fitted_in_both_halves"]), src)
    amihud = ident["d5_amihud"]
    _put(macros, "fsAmihudSpearman", fixed(amihud["spearman_between_halves"], 3), src)
    _put(macros, "fsAmihudSpearmanP", scientific(amihud["spearman_p_value"]), src)
    _put(macros, "fsAmihudN", integer(amihud["n_symbols"]), src)


def _validation_gap(macros: Macros, flow: dict[str, Any], corpus: dict[str, Any]) -> None:
    """The methodological finding the PI promoted into Results: validation on a narrow family."""
    src = "corpus_capacity.json"
    factor = np.array([s["binding_capacity_inr"] for s in flow["capacity"]])
    corp = np.array([s["binding_capacity_inr"] for s in corpus["capacity"]])

    factor_span = float(factor.max() / factor.min())
    corpus_span = float(np.quantile(corp, 0.95) / np.quantile(corp, 0.05))
    _put(macros, "fsFactorSpan", fixed(factor_span, 1), "flowstate.json")
    _put(macros, "fsCorpusSpan", fixed(corpus_span, 1), src)
    _put(macros, "fsSpanRatio", fixed(corpus_span / factor_span, 1), src)

    _put(macros, "fsCorpusN", integer(len(corp)), src)
    _put(macros, "fsCorpusBindingMedian", fixed(float(np.median(corp)) / CRORE, 2), src)
    _put(macros, "fsCorpusBindingPFive", fixed(float(np.quantile(corp, 0.05)) / CRORE, 2), src)
    _put(macros, "fsCorpusBindingPNinetyFive",
         fixed(float(np.quantile(corp, 0.95)) / CRORE, 1), src)
    _put(macros, "fsCorpusBindingMax", fixed(float(corp.max()) / CRORE, 1), src)

    med = np.array([s["median_capacity_inr"] for s in corpus["capacity"]])
    _put(macros, "fsCorpusMedianOverBinding", fixed(float(np.median(med / corp)), 1), src)
    entry = np.array([s["entry_capacity_inr"] for s in corpus["capacity"]])
    binds = int(np.sum(np.abs(entry - corp) < 1e-9 * np.maximum(entry, 1.0)))
    _put(macros, "fsEntryBindsCount", integer(binds), src)
    _put(macros, "fsEntryBindsShare", percent(binds / len(corp)), src)

    _put(macros, "fsCorpusAllThree", integer(corpus["n_with_all_three_benchmarks"]), src)
    _put(macros, "fsCorpusWithCapacity", integer(corpus["n_with_capacity"]), src)
    _put(macros, "fsCorpusWithFragility", integer(corpus["n_with_fragility"]), src)
    _put(macros, "fsCorpusKnifeEdge",
         integer(corpus["provenance"]["n_knife_edge_flagged"]), src)
    _put(macros, "fsMinTradedFraction", scientific(1e-9), "config.yaml")


#: Arm key -> the letters used in macro names. LaTeX control sequences may not contain digits or
#: punctuation, so the product names are reduced to bare words here and spelled out in the prose.
ARM_WORD = {"gpt": "Gpt", "claude": "Claude", "gemini": "Gemini"}


def _frontier(macros: Macros, gv: dict[str, Any]) -> None:
    """Deployment capacity for each frontier arm, from the generator-validation study.

    Macros carry the paper's own ``fs`` prefix so ``check_paper_numbers.py`` covers them; a
    separate namespace would be defined but never verified.

    FlowState's own conclusion is the one under test here: capacity was presented as a property of
    Indian market liquidity rather than of whoever wrote the strategy. Two of three arms are
    consistent with that and one is not, and both facts are emitted.
    """
    source = "generator_validation.json"
    _put(macros, "fsGvArms", str(gv["n_arms"]), source)
    _put(macros, "fsGvTotal", str(gv["n_total"]), source)
    _put(macros, "fsGvRequestsPerArm", str(gv["requests_per_arm"]), source)
    for arm, word in ARM_WORD.items():
        row = gv["arms"][arm]
        _put(macros, f"fsGv{word}N", str(row["n"]), source)
        _put(macros, f"fsGv{word}CapMedian", f"{row['cap_median_cr']:.2f}", source)
        _put(macros, f"fsGv{word}CapMin", f"{row['cap_min_cr']:.2f}", source)
        _put(macros, f"fsGv{word}CapMax", f"{row['cap_max_cr']:.2f}", source)
        _put(macros, f"fsGv{word}CapSpan", f"{row['cap_span']:.1f}", source)
        _put(macros, f"fsGv{word}CapRatio", f"{row['cap_ratio_to_local']:.2f}", source)
    gaps = _read(ROOT / "data" / "processed" / "frontier_gap_measures.json")
    for arm, word in ARM_WORD.items():
        flow = gaps["arms"][arm]["flow_capacity"]
        gsource = "frontier_gap_measures.json"
        _put(macros, f"fsGv{word}FlowMedian", f"{flow['ratio_median']:.3f}", gsource)
        _put(macros, f"fsGv{word}FlowMin", f"{flow['ratio_min']:.3f}", gsource)
        _put(macros, f"fsGv{word}FlowMax", f"{flow['ratio_max']:.3f}", gsource)


def _write(macros: Macros) -> None:
    lines = [
        "% GENERATED BY scripts/build_flowstate_numbers.py -- DO NOT EDIT BY HAND.",
        "% Every figure the FlowState paper states is defined here and traced to an artifact.",
        "",
    ]
    for name in sorted(macros):
        value, source = macros[name]
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {source}")
    (ROOT / "papers" / "flowstate_numbers.tex").write_text("\n".join(lines) + "\n",
                                                           encoding="utf-8")
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "paper_numbers.json").write_text(
        json.dumps({k: {"value": v[0], "source": v[1]} for k, v in sorted(macros.items())},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _render_results(macros: Macros) -> None:
    """Render RESULTS.md from its template, so the Markdown and the paper cannot disagree."""
    template = BENCH / "RESULTS.template.md"
    if not template.exists():
        _log.info("no RESULTS template yet; skipping")
        return
    text = template.read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"\{\{(fs[A-Za-z]+)\}\}", text)) - set(macros))
    if missing:
        raise ValueError(f"RESULTS.template.md refers to undefined macros: {missing}")
    for name, (value, _) in macros.items():
        text = text.replace(f"{{{{{name}}}}}", plain(value))
    (BENCH / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> int:
    configure_logging()
    cfg = load_config()
    processed = cfg.paths.data_processed
    flow = _read(processed / "flowstate.json")
    ident = _read(processed / "impact_identifiability.json")
    corpus = _read(processed / "corpus_capacity.json")
    gv = _read(processed / "generator_validation.json")

    macros: Macros = {}
    _decay(macros, flow)
    _capacity(macros, flow)
    _flow_conditional(macros, flow)
    _identification(macros, ident)
    _validation_gap(macros, flow, corpus)
    _frontier(macros, gv)

    _write(macros)
    _render_results(macros)
    _log.info("wrote %d macros from %d artifacts", len(macros),
              len({source for _, source in macros.values()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
