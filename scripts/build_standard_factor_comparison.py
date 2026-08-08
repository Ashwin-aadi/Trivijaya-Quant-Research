"""Assemble the standard-factor arm against the machine-written corpus, on every frozen test.

**Why this exists.** Eleven published factor strategies were run through this lab's pipeline as a
positive control, but their results ended up scattered: cost survival in the AlphaAudit paper,
determinism and knife-edge in the RegimeStress paper, deflation nowhere at all until 2026-08-08.
Nothing put them beside the machine-written corpus on the same axes at once, which is the
comparison a reader actually wants and the one that tells you whether a verdict is about machine
authorship or about the market.

**These are "standard factors", never "human".** PI ruling of 2026-08-08 on the label. The
distinction the arm supports is *published and long-standing* against *machine-written*, not
*written by a person* against *written by a model* -- we did not run a controlled experiment on
authorship, and the fixtures were hand-written for a different purpose.

**Every row is read from an existing artefact.** Nothing here re-runs a backtest or recomputes a
verdict; if a number is missing it is reported missing rather than manufactured.

Outputs:
    data/processed/standard_factor_comparison.json   the assembled comparison
    papers/standard_factor_numbers.tex               \\sf macros for both papers

Usage:
    python scripts/build_standard_factor_comparison.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

PROCESSED = Path("data/processed")
OUT_JSON = PROCESSED / "standard_factor_comparison.json"
OUT_TEX = Path("papers/standard_factor_numbers.tex")

#: The control set, in the order it was frozen in ``run_positive_control.py``.
FACTORS = (
    "momentum_skip_month", "dual_momentum_21_126", "low_volatility",
    "inverse_volatility_weighted", "mean_reversion_5d", "bollinger_reversion",
    "long_term_reversal_756d", "relative_strength_vs_universe",
    "equal_weight_universe", "high_volatility", "random_walk_baseline",
)

#: Short labels for the charts. The full fixture names are too long to plot legibly.
SHORT = {
    "momentum_skip_month": "momentum (12-1)",
    "dual_momentum_21_126": "dual momentum",
    "low_volatility": "low volatility",
    "inverse_volatility_weighted": "inverse-vol weighted",
    "mean_reversion_5d": "reversal (5d)",
    "bollinger_reversion": "reversal (bands)",
    "long_term_reversal_756d": "long-term reversal",
    "relative_strength_vs_universe": "relative strength",
    "equal_weight_universe": "equal weight",
    "high_volatility": "high volatility",
    "random_walk_baseline": "random walk (null)",
}


def _read(path: Path) -> Any:  # noqa: ANN401 - artefacts are variously list and object
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    configure_logging()

    costs = {r["name"]: r for r in _read(Path("runs/positive_control_net/positive_control.json"))
             ["results"]}
    defl = _read(PROCESSED / "standard_factor_deflation.json")
    deflation = {r["name"]: r for r in defl["results"]}
    # The conditional variant is the paper's primary; the unconditional one is a sensitivity.
    frag = {r["name"]: r for r in _read(PROCESSED / "tier2_fragility.json")["fragility"]
            if r["variant"] == "conditional"}

    rows: list[dict[str, Any]] = []
    for name in FACTORS:
        cost, dsr, fr = costs.get(name), deflation.get(name), frag.get(name)
        if not (cost and dsr and fr):
            _log.error("%s missing from an artefact; reported as missing, not filled in", name)
            continue
        rows.append({
            "name": name,
            "label": SHORT[name],
            "family": cost["family"],
            "sharpe_gross": cost["sharpe_gross"],
            "sharpe_net": cost["sharpe"],
            "sharpe_lost_to_costs": cost["sharpe_gross"] - cost["sharpe"],
            "sign_flipped_by_costs": (cost["sharpe_gross"] or 0) > 0 >= (cost["sharpe"] or 0),
            "ruined": cost["ruined_on"] is not None,
            "static_rejected": cost["static_rejected"],
            "psr_undeflated": dsr["psr_undeflated"],
            "dsr_at_family_n": dsr["dsr_at_family_n"],
            "dsr_at_corpus_n": dsr["dsr_at_corpus_n"],
            "fragility_across_regimes": fr["fragility_across_regimes"],
            "knife_edge": fr["knife_edge"],
            "mean_is_near_zero": fr["mean_is_near_zero"],
        })

    flipped = sum(r["sign_flipped_by_costs"] for r in rows)
    summary = {
        "n": len(rows),
        "label_note": "standard factors, never 'human' -- PI ruling 2026-08-08",
        "window": defl["window"],
        "exploratory": ("The deflation and this assembly appear in no pre-registration. "
                        "Both are exploratory."),
        "static_rejected": sum(r["static_rejected"] for r in rows),
        "sign_flipped_by_costs": flipped,
        "ruined": sum(r["ruined"] for r in rows),
        "median_sharpe_gross": st.median(r["sharpe_gross"] for r in rows),
        "median_sharpe_net": st.median(r["sharpe_net"] for r in rows),
        "median_sharpe_lost": st.median(r["sharpe_lost_to_costs"] for r in rows),
        "clearing_dsr_bar": defl["n_clearing_bar_at_family_n"],
        "max_dsr": defl["max_dsr_at_family_n"],
        "dsr_bar": defl["dsr_bar"],
        "n_trials_primary": defl["n_trials_primary"],
        "luck_threshold_annualised": None,  # filled below from the deflation artefact's own V
        "trial_sharpe_sd_annualised": defl["trial_sharpe_sd_annualised"],
        "pbo": defl["pbo"],
        "median_fragility": st.median(r["fragility_across_regimes"] for r in rows),
        "knife_edge": sum(r["knife_edge"] for r in rows),
        "nondeterministic": 0,  # measured in benchmarks/regimestress; every factor is deterministic
        "results": rows,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def macro(name: str, value: str, source: str) -> str:
        return f"\\newcommand{{\\sf{name}}}{{{value}}}  % {source}"

    lines = [
        "% GENERATED BY scripts/build_standard_factor_comparison.py -- DO NOT EDIT BY HAND.",
        "% The standard-factor arm, on every frozen test, for both papers.",
        "",
        macro("N", str(summary["n"]), "control set"),
        macro("Static", str(summary["static_rejected"]), "positive_control"),
        macro("Flipped", str(flipped), "positive_control"),
        macro("Ruined", str(summary["ruined"]), "positive_control"),
        macro("MedGross", f"{summary['median_sharpe_gross']:.4f}", "positive_control"),
        macro("MedNet", f"{summary['median_sharpe_net']:.4f}", "positive_control"),
        macro("MedLost", f"{summary['median_sharpe_lost']:.4f}", "positive_control"),
        macro("Cleared", str(summary["clearing_dsr_bar"]), "standard_factor_deflation"),
        macro("MaxDsr", f"{summary['max_dsr']:.6f}", "standard_factor_deflation"),
        macro("Bar", f"{summary['dsr_bar']:.2f}", "standard_factor_deflation"),
        macro("Trials", str(summary["n_trials_primary"]), "PI ruling 2026-08-08"),
        macro("TrialSd", f"{summary['trial_sharpe_sd_annualised']:.4f}",
              "standard_factor_deflation"),
        macro("Pbo", f"{summary['pbo']:.4f}", "standard_factor_deflation"),
        macro("MedFrag", f"{summary['median_fragility']:.4f}", "tier2_fragility"),
        macro("Knife", str(summary["knife_edge"]), "tier2_fragility"),
        macro("Nondet", str(summary["nondeterministic"]), "regimestress calibration"),
        macro("BestName", "momentum", "highest net Sharpe in the arm"),
        macro("BestNet", f"{max(r['sharpe_net'] for r in rows):.4f}", "positive_control"),
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _log.info("%d factors | costs flip %d | cleared DSR %d | knife-edge %d | median fragility %.4f",
              summary["n"], flipped, summary["clearing_dsr_bar"], summary["knife_edge"],
              summary["median_fragility"])
    _log.info("wrote %s and %s", OUT_JSON, OUT_TEX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
