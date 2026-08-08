"""The RULE 11 control: plain prompting given each arm's budget, and allowed to keep its best.

**This is the primary comparison of P4.** A paradigm that spends five times the tokens of plain
prompting is not better for beating it; it is a different point on a budget curve. The control is
therefore never G1 at its natural budget -- it is G1 run `k` times at the treatment's budget with
best-of-`k` selection, and no arm may be reported as a win against anything else.

The design is PREREGISTRATION.md §"The compute-matched control", implemented here without deviation:

1. `tokens_per_accepted` is **generated** output tokens per accepted draw, from Amendment 3 §3.1's
   committed table. Not prompt tokens, not calls -- a call-count ratio would hand the multi-call
   paradigms a free budget increase.
2. `k = ceil(tokens_per_accepted(T) / tokens_per_accepted(G1))`, rounded **up**, so the rounding
   surplus goes to the control rather than to the treatment.
3. `k` **consecutive** indices are drawn from P1's 1,550-candidate corpus and the best kept by
   development Sharpe.

**Blocks are contiguous, never a uniform random sample.** P1's corpus is stratified --
``theme_for(index)`` is ``THEMES[index % 12]`` -- so `k` consecutive indices cover `k` consecutive
themes exactly as `k` fresh draws would. A uniform sample draws a random theme mixture with repeats:
a different sampling scheme, a different variance, and a confound invisible in the output.

**A barren block counts.** Only a minority of P1's candidates executed and took a position, so for
small `k` many blocks contain nothing rankable. They count in the control's yield denominator.
Discarding them would compare the control's best blocks against every one of the treatment's,
inflating the control and manufacturing a null -- which, being the tidy result, is the one that most
needs guarding against.

Writes `benchmarks/generationbench/compute_matched_control.json`.

Usage:
    python scripts/paradigm_compute_matched_control.py
"""

from __future__ import annotations

import json
import sys
from math import ceil
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")
P1_RESULTS = Path("runs/pooled/backtest_results.json")
OUT = Path("benchmarks/generationbench/compute_matched_control.json")

# Amendment 3 section 3.1, the committed realised design. summary.json holds only the final sitting
# of each arm and cannot be used for this: it reports 330 of G1's 830 draws.
GENERATED = {
    "G1": {"draws": 830, "output_tokens": 407588},
    "G2": {"draws": 800, "output_tokens": 408215},
    "G4": {"draws": 365, "output_tokens": 409161},
    "G5": {"draws": 288, "output_tokens": 408324},
    "G6": {"draws": 152, "output_tokens": 410177},
    "G7": {"draws": 60, "output_tokens": 408053},
}
TREATMENTS = ("G2", "G4", "G5", "G6", "G7")


def tokens_per_accepted(arm: str) -> float:
    entry = GENERATED[arm]
    return entry["output_tokens"] / entry["draws"]


def arm_sharpes(arm: str) -> list[float]:
    """Development Sharpe for every candidate in `arm` that executed and took a position."""
    rows = json.loads((CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))
    return [row["sharpe"] for row in rows
            if row["outcome"] == "evaluated" and (row.get("mean_turnover") or 0) > 0
            and row["sharpe"] is not None]


def control_blocks(p1: list[dict[str, Any]], k: int) -> tuple[list[float], int]:
    """Best development Sharpe of each contiguous block of `k`, and the count of barren blocks.

    Blocks are non-overlapping and taken in corpus order. A block with nothing rankable contributes
    no Sharpe but is counted, because it is a draw the control spent its budget on.
    """
    best: list[float] = []
    barren = 0
    for start in range(0, len(p1) - k + 1, k):
        block = p1[start:start + k]
        rankable = [row["sharpe"] for row in block
                    if row["outcome"] == "evaluated" and (row.get("mean_turnover") or 0) > 0
                    and row["sharpe"] is not None]
        if rankable:
            best.append(max(rankable))
        else:
            barren += 1
    return best, barren


def main() -> int:
    configure_logging()
    p1 = json.loads(P1_RESULTS.read_text(encoding="utf-8"))
    _log.info("P1 corpus: %d candidates; G1 costs %.1f generated tokens per accepted draw",
              len(p1), tokens_per_accepted("G1"))

    records = []
    for arm in TREATMENTS:
        ratio = tokens_per_accepted(arm) / tokens_per_accepted("G1")
        k = ceil(ratio)
        best, barren = control_blocks(p1, k)
        treatment = arm_sharpes(arm)

        # Yield is over blocks for the control and over draws for the treatment: both are "what one
        # unit of the same budget produced", which is the quantity RULE 11 makes comparable.
        n_blocks = len(best) + barren
        control_yield = len(best) / n_blocks if n_blocks else float("nan")
        treatment_yield = len(treatment) / GENERATED[arm]["draws"]

        record = {
            "arm": arm,
            "tokens_per_accepted": tokens_per_accepted(arm),
            "ratio_to_g1": ratio,
            "k": k,
            "control_n_blocks": n_blocks,
            "control_n_barren": barren,
            "control_yield": control_yield,
            "control_median_best_sharpe": float(np.median(best)) if best else float("nan"),
            "control_mean_best_sharpe": float(np.mean(best)) if best else float("nan"),
            "control_max_best_sharpe": float(max(best)) if best else float("nan"),
            "treatment_n_draws": GENERATED[arm]["draws"],
            "treatment_n_rankable": len(treatment),
            "treatment_yield": treatment_yield,
            "treatment_median_sharpe": float(np.median(treatment)) if treatment else float("nan"),
            "treatment_max_sharpe": float(max(treatment)) if treatment else float("nan"),
            "treatment_beats_control_on_median": (
                float(np.median(treatment)) > float(np.median(best))
                if treatment and best else None),
        }
        records.append(record)
        _log.info("%-3s k=%2d | control %3d blocks, %3d barren, yield %5.1f%%, median best "
                  "%6.3f | treatment yield %5.1f%%, median %6.3f",
                  arm, k, n_blocks, barren, control_yield * 100,
                  record["control_median_best_sharpe"], treatment_yield * 100,
                  record["treatment_median_sharpe"])

    OUT.write_text(json.dumps({
        "rule": "RULE 11, implemented per PREREGISTRATION.md 'The compute-matched control'",
        "matching_unit": "generated output tokens per accepted draw",
        "control_corpus": str(P1_RESULTS),
        "control_corpus_n": len(p1),
        "note": ("Barren blocks are counted in the control's yield denominator. Blocks are "
                 "contiguous, preserving P1's index-modulo-12 theme stratification."),
        "arms": records,
    }, indent=2, sort_keys=True), encoding="utf-8")
    _log.info("written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
