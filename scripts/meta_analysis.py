"""Every number this lab can compute about generated strategies, in one pass, from the artefacts.

**Purpose.** P1 measured one generator under one prompting method. Since then the same frozen stack
has been pointed at three frontier models and six generation paradigms. This assembles all of it and
computes the cross-cutting comparisons no single project could make, so the papers can be rewritten
around what is now known rather than around what P1 could see.

**Nothing here is transcribed.** Every figure is read from a committed artefact and recomputed. If
an artefact is missing, the block reports it missing rather than falling back on a remembered value.

## The two axes, which must not be confused

- **The generator axis.** Identical frozen task specification, identical funnel, different model:
  local 7B against GPT, Claude and Gemini. This asks whether the benchmarks measure a property of
  machine-written strategies or a peculiarity of one small local model.
- **The methodology axis.** Identical model, identical funnel, different prompting structure: six
  paradigms at matched generated tokens. This asks how an AI system should generate strategies.

**The generator axis is NOT compute-matched and the methodology axis is.** RULE 11 governs the
second. The first compares 20 frontier draws against 830 local ones at whatever each cost, so every
generator-axis figure is a *rate* claim and none of them may be reported as a compute-matched win.
This is stated in the output so it cannot be dropped in transcription.

## Comparability hazards, recorded in the output rather than in a footnote

1. **Fragility is measured at different tiers.** The frontier arms carry tier 1 across-paths
   fragility at 100 synthetic price panels; P4 carries tier 2 across-regimes at 1,000 resampled
   return paths. P2 measured tier agreement on fragility at Spearman 0.620 and published that tier
   2 cannot substitute for tier 1 when fragility is the quantity of interest. **The two are reported
   in separate blocks and never in one table.**
2. **Sample sizes differ by up to 40x.** 20 frontier draws against 830 for G1.
3. **Capacity for P4 uses the unfiltered price panel; the frontier arms use the truncated one**
   (FlowState CORRECTIONS.md C7). Capacity appeared insensitive to this on one arm; it is not
   asserted for the others.

Writes `benchmarks/generationbench/META.json`.

Usage:
    python scripts/meta_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")
FRONTIER = Path("runs")
OUT = Path("benchmarks/generationbench/META.json")

PARADIGMS = {
    "G1": "plain prompting",
    "G2": "chain-of-thought",
    "G4": "planning",
    "G5": "reflection",
    "G6": "graph-of-thoughts",
    "G7": "Monte Carlo tree search",
}
FRONTIER_ARMS = {
    "frontier_gpt": "GPT",
    "frontier_claude": "Claude",
    "frontier_gemini": "Gemini",
}


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _sharpes(rows: list[dict[str, Any]], *, traded_only: bool) -> list[float]:
    """Sharpe for every row that executed, optionally restricted to those taking a position."""
    out = []
    for row in rows:
        if row["outcome"] != "evaluated" or row.get("sharpe") is None:
            continue
        if traded_only and not (row.get("mean_turnover") or 0) > 0:
            continue
        out.append(float(row["sharpe"]))
    return out


def funnel_rates(bt: list[dict[str, Any]], ho: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Front-of-funnel rates and back-of-funnel performance for one corpus.

    "Front" is whether the model produced something that runs and trades at all -- a property of
    the code it wrote. "Back" is whether what it wrote was any good out of sample. Separating them
    is the point: a generator can be far better at one and no better at the other.
    """
    n = len(bt)
    executed = [r for r in bt if r["outcome"] == "evaluated"]
    traded = [r for r in executed if (r.get("mean_turnover") or 0) > 0]
    dev = _sharpes(bt, traded_only=True)
    hold = _sharpes(ho, traded_only=True) if ho else []
    return {
        "n_draws": n,
        "n_executed": len(executed),
        "execution_rate": len(executed) / n if n else None,
        "n_traded": len(traded),
        "position_taking_rate": len(traded) / n if n else None,
        "dev_median_sharpe": _median(dev),
        "dev_max_sharpe": max(dev) if dev else None,
        "holdout_n": len(hold),
        "holdout_median_sharpe": _median(hold),
        "holdout_max_sharpe": max(hold) if hold else None,
        "holdout_fraction_positive": (
            sum(1 for s in hold if s > 0) / len(hold) if hold else None),
    }


def audit_rates(audit: dict[str, Any]) -> dict[str, Any]:
    """Static and semantic rejection rates, and the statistical layer's saturation."""
    static = audit.get("static", {})
    semantic = audit.get("semantic", {})
    statistical = audit.get("statistical", {})
    srej = sum(1 for v in static.values() if v.get("rejected"))
    mrej = sum(1 for v in semantic.values() if v.get("rejected"))
    trej = sum(1 for v in statistical.values() if v.get("rejected"))
    return {
        "static_n": len(static), "static_rejected": srej,
        "static_rejection_rate": srej / len(static) if static else None,
        "semantic_n": len(semantic), "semantic_rejected": mrej,
        "semantic_rejection_rate": mrej / len(semantic) if semantic else None,
        "statistical_n": len(statistical), "statistical_rejected": trej,
        "statistical_rejects_everything": bool(statistical) and trej == len(statistical),
    }


def ablation_block(path: Path) -> dict[str, Any] | None:
    """AUAP for every layer combination, and whether any beat random rejection."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    combos = payload["combinations"]
    best = max(combos, key=lambda c: c["auap"])
    lo, hi = payload["random_baseline_auap_interval"]
    return {
        "n_ranked": payload["n_candidates"],
        "reportable": bool(payload.get("reportable_auap")),
        "baseline_interval": [lo, hi],
        "n_combinations": len(combos),
        "n_beating_random": sum(1 for c in combos if c["beats_random"]),
        "best_layers": best["layers"],
        "best_auap": best["auap"],
        "best_p_at_005": best["p_at_005"],
    }


def capacity_block(payload: dict[str, Any], near_cash: set[str]) -> dict[str, Any]:
    """Constraint capacity in crore, split by whether the strategy deployed its capital."""
    deployed = [s["binding_capacity_inr"] / 1e7 for s in payload["capacity"]
                if s["factor"] not in near_cash]
    cash = [s["binding_capacity_inr"] / 1e7 for s in payload["capacity"]
            if s["factor"] in near_cash]
    return {
        "measure": "constraint-based deployment capacity, never impact erosion",
        "deployed_n": len(deployed),
        "deployed_median_crore": _median(deployed),
        "deployed_max_crore": max(deployed) if deployed else None,
        "near_cash_n": len(cash),
        "near_cash_median_crore": _median(cash),
    }


def paradigm_axis() -> list[dict[str, Any]]:
    """One record per generation paradigm: the methodology axis, compute-matched by design."""
    control = json.loads(
        (Path("benchmarks/generationbench/compute_matched_control.json")).read_text(
            encoding="utf-8"))
    by_arm = {r["arm"]: r for r in control["arms"]}
    funnel = json.loads(
        (Path("benchmarks/generationbench/funnel.json")).read_text(encoding="utf-8"))
    survival = {r["arm"]: r for r in funnel["views"]["equal_token"]}
    redundancy = json.loads(
        (Path("benchmarks/generationbench/redundancy.json")).read_text(encoding="utf-8"))["arms"]
    power = {r["arm"]: r for r in json.loads(
        (Path("benchmarks/generationbench/redundancy_power.json")).read_text(
            encoding="utf-8"))["arms"]}
    cross = json.loads(
        (Path("benchmarks/generationbench/cross_arm_duplicates.json")).read_text(
            encoding="utf-8"))

    records = []
    for arm, label in PARADIGMS.items():
        bt = json.loads((CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))
        ho_path = CORPUS / arm / "holdout_results.json"
        ho = json.loads(ho_path.read_text(encoding="utf-8")) if ho_path.exists() else None
        audit = json.loads((CORPUS / arm / "audit_results.json").read_text(encoding="utf-8"))
        exposure = json.loads((CORPUS / arm / "exposure.json").read_text(encoding="utf-8"))
        near_cash = {r["name"] for r in exposure["exposure"] if r["near_cash"]}
        capacity = json.loads((CORPUS / arm / "capacity.json").read_text(encoding="utf-8"))
        frag = json.loads((CORPUS / arm / "fragility.json").read_text(encoding="utf-8"))
        clean = [r["fragility_across_regimes"] for r in frag["fragility"]
                 if r["variant"] == "conditional" and not r["knife_edge"]]
        calib = json.loads((CORPUS / arm / "calibration.json").read_text(encoding="utf-8"))

        traded = redundancy[arm]["n_traded"]
        record = {
            "arm": arm, "paradigm": label,
            "funnel": funnel_rates(bt, ho),
            "audit": audit_rates(audit),
            "trials_arm": audit["n_trials_arm"],
            "pbo": audit["pbo"],
            "full_stack_survival": survival[arm]["full_stack_survival"],
            "compute_matched": {
                "k": by_arm[arm]["k"] if arm in by_arm else None,
                "control_yield": by_arm[arm]["control_yield"] if arm in by_arm else None,
                "treatment_yield": by_arm[arm]["treatment_yield"] if arm in by_arm else None,
                "beats_control_on_yield": (
                    by_arm[arm]["treatment_yield"] > by_arm[arm]["control_yield"]
                    if arm in by_arm else None),
            },
            "redundancy_of_traded": redundancy[arm]["redundancy_of_traded"],
            "p_zero_duplicates_at_reference_rate": power[arm][
                "p_zero_duplicates_at_reference_rate"],
            "cross_arm_duplicate_share": (
                cross["per_arm_in_cross_arm_cluster"].get(arm, 0)
                / cross["per_arm_traded"][arm] if cross["per_arm_traded"].get(arm) else None),
            "knife_edge_rate": len(calib.get("knife_edge", [])) / traded if traded else None,
            "fragility_tier2_across_regimes_median": _median(clean),
            "fragility_n": len(clean),
            "capacity": capacity_block(capacity, near_cash),
            "ablation_holdout": ablation_block(CORPUS / arm / "ablation_holdout.json"),
        }
        records.append(record)
    return records


def generator_axis() -> list[dict[str, Any]]:
    """One record per frontier model, plus the local reference. NOT compute-matched."""
    records = []
    # The local reference is G1: the same frozen task specification under plain prompting, which is
    # what the frontier arms were given. Comparing them to a scaffolded arm would confound the axes.
    bt = json.loads((CORPUS / "G1" / "backtest_results.json").read_text(encoding="utf-8"))
    ho = json.loads((CORPUS / "G1" / "holdout_results.json").read_text(encoding="utf-8"))
    audit = json.loads((CORPUS / "G1" / "audit_results.json").read_text(encoding="utf-8"))
    records.append({
        "generator": "local 7B (qwen2.5:7b-instruct-q4_K_M)",
        "role": "reference",
        "funnel": funnel_rates(bt, ho),
        "audit": audit_rates(audit),
        "ablation_holdout": ablation_block(CORPUS / "G1" / "ablation_holdout.json"),
        "fragility_tier": 2,
    })

    for arm, label in FRONTIER_ARMS.items():
        d = FRONTIER / arm
        if not (d / "backtest_results.json").exists():
            _log.warning("%s missing backtest results; skipped", arm)
            continue
        bt = json.loads((d / "backtest_results.json").read_text(encoding="utf-8"))
        ho_path = d / "holdout_results.json"
        ho = json.loads(ho_path.read_text(encoding="utf-8")) if ho_path.exists() else None
        audit = json.loads((d / "audit_results.json").read_text(encoding="utf-8"))
        frag_path = d / "fragility.json"
        frag = json.loads(frag_path.read_text(encoding="utf-8")) if frag_path.exists() else {}
        across_paths = [v["fragility_across_paths"] for v in frag.values()
                        if isinstance(v, dict) and "fragility_across_paths" in v]
        cap_path = d / "capacity.json"
        capacity = (capacity_block(json.loads(cap_path.read_text(encoding="utf-8")), set())
                    if cap_path.exists() else None)
        records.append({
            "generator": label,
            "role": "frontier arm",
            "funnel": funnel_rates(bt, ho),
            "audit": audit_rates(audit),
            "ablation_holdout": ablation_block(d / "ablation_holdout.json"),
            "fragility_tier": 1,
            "fragility_tier1_across_paths_median": _median(across_paths),
            "fragility_n": len(across_paths),
            "capacity": capacity,
        })
    return records


def auditor_null(paradigms: list[dict[str, Any]],
                 generators: list[dict[str, Any]]) -> dict[str, Any]:
    """How many independent corpora the abstention null now survives, and how many combinations.

    This is the single strongest claim the programme has, because it is the one that could most
    easily have failed: a metric that beat random on one corpus and not another would be evidence
    about the corpora, not about the auditor.
    """
    blocks = []
    for record in paradigms:
        if record["ablation_holdout"]:
            blocks.append((f"paradigm {record['arm']}", record["ablation_holdout"]))
    for record in generators:
        if record["role"] == "frontier arm" and record["ablation_holdout"]:
            blocks.append((f"generator {record['generator']}", record["ablation_holdout"]))

    p1 = ablation_block(Path("runs/pooled/ablation_holdout.json"))
    if p1:
        blocks.append(("AlphaAudit pooled corpus", p1))

    total = sum(b["n_combinations"] for _, b in blocks)
    beating = sum(b["n_beating_random"] for _, b in blocks)
    return {
        "n_corpora": len(blocks),
        "n_combinations_total": total,
        "n_combinations_beating_random": beating,
        "all_reportable": all(b["reportable"] for _, b in blocks),
        "corpora": [{"corpus": name, "n_ranked": b["n_ranked"],
                     "beats": b["n_beating_random"], "of": b["n_combinations"],
                     "best_auap": b["best_auap"]} for name, b in blocks],
    }


def front_vs_back(generators: list[dict[str, Any]]) -> dict[str, Any]:
    """Does frontier capability move the front of the funnel, the back, or both?

    Front: execution and position-taking rates, properties of the code the model wrote.
    Back: out-of-sample Sharpe and whether the auditor's ranking becomes informative.
    """
    ref = next(r for r in generators if r["role"] == "reference")
    arms = [r for r in generators if r["role"] == "frontier arm"]
    if not arms:
        return {"available": False}

    return {
        "available": True,
        "not_compute_matched": True,
        "reference": ref["generator"],
        "front": {
            "reference_execution_rate": ref["funnel"]["execution_rate"],
            "reference_position_taking_rate": ref["funnel"]["position_taking_rate"],
            "frontier_execution_rates": {
                r["generator"]: r["funnel"]["execution_rate"] for r in arms},
            "frontier_position_taking_rates": {
                r["generator"]: r["funnel"]["position_taking_rate"] for r in arms},
            "every_frontier_arm_exceeds_reference": all(
                r["funnel"]["position_taking_rate"] > ref["funnel"]["position_taking_rate"]
                for r in arms),
        },
        "back": {
            "reference_holdout_median_sharpe": ref["funnel"]["holdout_median_sharpe"],
            "frontier_holdout_median_sharpe": {
                r["generator"]: r["funnel"]["holdout_median_sharpe"] for r in arms},
            "any_frontier_arm_better_out_of_sample": any(
                (r["funnel"]["holdout_median_sharpe"] or -9e9)
                > (ref["funnel"]["holdout_median_sharpe"] or -9e9) for r in arms),
            "frontier_combinations_beating_random": sum(
                r["ablation_holdout"]["n_beating_random"] for r in arms
                if r["ablation_holdout"]),
        },
    }


def sophistication(paradigms: list[dict[str, Any]]) -> dict[str, Any]:
    """Does prompting sophistication survive a compute-matched control?"""
    scaffolded = [r for r in paradigms if r["arm"] != "G1"]
    return {
        "n_scaffolded_arms": len(scaffolded),
        "n_beating_control_on_yield": sum(
            1 for r in scaffolded if r["compute_matched"]["beats_control_on_yield"]),
        "per_arm": {r["arm"]: {
            "k": r["compute_matched"]["k"],
            "control_yield": r["compute_matched"]["control_yield"],
            "treatment_yield": r["compute_matched"]["treatment_yield"],
        } for r in scaffolded},
        "claim_sanctioned_by_pi": (
            "Under compute-matched evaluation, no tested scaffolded paradigm improved yield over "
            "plain prompting."),
        "claim_explicitly_not_made": (
            "Scaffolding makes performance worse. The experiment does not establish this."),
    }


def redundancy_power_note(paradigms: list[dict[str, Any]]) -> dict[str, Any]:
    """Which arms' redundancy figures have the power to mean anything."""
    return {
        "status": "EXPLORATORY -- not pre-registered",
        "per_arm": {r["arm"]: {
            "redundancy_of_traded": r["redundancy_of_traded"],
            "p_zero_at_reference_rate": r["p_zero_duplicates_at_reference_rate"],
            "informative": r["p_zero_duplicates_at_reference_rate"] < 0.05,
        } for r in paradigms},
        "n_arms_with_informative_zero": sum(
            1 for r in paradigms
            if r["redundancy_of_traded"] == 0
            and r["p_zero_duplicates_at_reference_rate"] < 0.05),
    }


def main() -> int:
    configure_logging()
    paradigms = paradigm_axis()
    generators = generator_axis()

    meta = {
        "purpose": (
            "Every computable number about machine-written strategies measured by this lab's "
            "frozen stack, across four generators and six generation paradigms."),
        "axes": {
            "methodology": "identical model, six paradigms, matched generated tokens (RULE 11)",
            "generator": ("identical prompt and funnel, four models, NOT compute-matched -- rate "
                          "claims only"),
        },
        "comparability_hazards": [
            "Fragility: frontier arms are tier 1 across-paths (100 panels); P4 is tier 2 "
            "across-regimes (1,000 return paths). P2 measured tier agreement on fragility at "
            "Spearman 0.620 and published that tier 2 cannot substitute for tier 1. Never tabled "
            "together.",
            "Sample sizes differ by up to 40x: 20 frontier draws against 830 for G1.",
            "P4 capacity uses the unfiltered price panel; frontier capacity uses the truncated one "
            "(FlowState CORRECTIONS.md C7).",
            "The generator axis is not compute-matched. No generator-axis figure is a RULE 11 win.",
        ],
        "methodology_axis": paradigms,
        "generator_axis": generators,
        "findings": {
            "sophistication_under_compute_matching": sophistication(paradigms),
            "frontier_front_vs_back": front_vs_back(generators),
            "auditor_abstention_null": auditor_null(paradigms, generators),
            "redundancy_power": redundancy_power_note(paradigms),
        },
    }
    OUT.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    null = meta["findings"]["auditor_abstention_null"]
    front = meta["findings"]["frontier_front_vs_back"]
    soph = meta["findings"]["sophistication_under_compute_matching"]
    _log.info("auditor null: %d of %d combinations beat random across %d corpora "
              "(all reportable: %s)",
              null["n_combinations_beating_random"], null["n_combinations_total"],
              null["n_corpora"], null["all_reportable"])
    _log.info("sophistication: %d of %d scaffolded arms beat their compute-matched control",
              soph["n_beating_control_on_yield"], soph["n_scaffolded_arms"])
    if front["available"]:
        _log.info("frontier front: every arm exceeds local position-taking = %s",
                  front["front"]["every_frontier_arm_exceeds_reference"])
        _log.info("frontier back: any arm better out of sample = %s; combinations beating "
                  "random = %d",
                  front["back"]["any_frontier_arm_better_out_of_sample"],
                  front["back"]["frontier_combinations_beating_random"])
    _log.info("written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
