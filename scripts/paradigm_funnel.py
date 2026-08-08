"""Survival through the whole frozen stack, per arm, at matched tokens and at matched n.

**RQ2 asks whether any paradigm raises the rate of strategies surviving the *full* stack**, as
opposed to any single stage. A paradigm can lead on static cleanliness and lose everything at
capacity, and reporting stages separately hides that. This walks every draw through the funnel in
order and reports what is left at each step.

The stages, and what each one drops:

1. **generated** -- every draw the arm produced.
2. **static** -- passes the P1 AST auditor.
3. **semantic** -- the local 7B finds the rationale consistent with the code. Read from
   `semantic_verdicts.jsonl`, which is the complete run; `audit_results.json` carries only a partial
   early pass and is not used.
4. **traded** -- executes and takes a position. This is where most of the corpus dies.
5. **stress** -- has a tier-2 fragility and is not knife-edge.
6. **deployed** -- holds a real book, i.e. is not near-cash by Amendment 4 §4.3.

**Two views, because they answer different questions.** The equal-token view is the primary design
and the one RULE 11 makes comparable. The equal-n view takes the first 60 draws of every arm -- 60
because G7, the most expensive arm, produced exactly that many -- and asks what the comparison would
have looked like under the fixed-n design Amendment 2 replaced. It is **secondary and exploratory**:
it discards 96% of G7's budget advantage over G1 and so cannot settle a compute question. It is
reported because the two views disagreeing would itself be worth knowing.

Writes `benchmarks/generationbench/funnel.json`.

Usage:
    python scripts/paradigm_funnel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")
OUT = Path("benchmarks/generationbench/funnel.json")
# G7, the most expensive arm, produced exactly 60 draws inside the shared token budget.
EQUAL_N = 60
STAGES = ("generated", "static", "semantic", "traded", "stress", "deployed")
# Reported beside the funnel, never inside it. See the comment in _funnel.
SATURATED = "statistical_if_applied"


def _load(arm: str) -> dict[str, Any]:
    """Every per-candidate verdict for one arm, keyed by candidate name."""
    audit = json.loads((CORPUS / arm / "audit_results.json").read_text(encoding="utf-8"))
    backtests = json.loads((CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))
    fragility = json.loads((CORPUS / arm / "fragility.json").read_text(encoding="utf-8"))
    exposure = json.loads((CORPUS / arm / "exposure.json").read_text(encoding="utf-8"))
    semantic = {}
    for line in (CORPUS / arm / "semantic_verdicts.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            semantic[row["name"]] = row

    return {
        "order": [row["name"] for row in backtests],
        "static": audit["static"],
        "statistical": audit["statistical"],
        "semantic": semantic,
        "traded": {row["name"] for row in backtests
                   if row["outcome"] == "evaluated" and (row.get("mean_turnover") or 0) > 0},
        # Conditional variant only, and knife-edge excluded, matching how fragility is reported.
        "stressed": {r["name"] for r in fragility["fragility"]
                     if r["variant"] == "conditional" and not r["knife_edge"]},
        "near_cash": {r["name"] for r in exposure["exposure"] if r["near_cash"]},
    }


def _funnel(data: dict[str, Any], names: list[str]) -> dict[str, int]:
    """Cumulative survivors at each stage, in order, over the given candidate names."""
    alive = list(names)
    counts = {"generated": len(alive)}

    alive = [n for n in alive if not data["static"].get(n, {}).get("rejected", False)]
    counts["static"] = len(alive)

    # A candidate with no semantic verdict is not silently passed: it is counted as not surviving,
    # so an incomplete stage shows up as attrition rather than as a free pass.
    alive = [n for n in alive
             if n in data["semantic"] and not data["semantic"][n]["rejected"]]
    counts["semantic"] = len(alive)

    alive = [n for n in alive if n in data["traded"]]
    counts["traded"] = len(alive)

    # Recorded, then **not** applied. The statistical layer rejects every candidate it scores, in
    # every arm, at a recorded confidence of exactly 1.0 -- which is P1's published behaviour
    # replicated, not a new defect: benchmarks/alphaaudit/RESULTS.md section 10 states that at
    # N=1,887 over 1,232 sessions "every evaluated strategy is rejected" and that "whether this
    # reflects appropriate statistical correction or saturation of the test cannot be determined
    # from these data". P4 pools 3,499 trials, so the term is larger still. A layer that rejects
    # everything carries no ordering information, and applying it here would make all six arms
    # identically zero and answer RQ2 with an artefact of the test rather than a property of the
    # paradigms.
    #
    # **This is a definitional choice made because it produces a non-empty answer, and it is
    # flagged as one** -- the same choice P1 made and flagged in its section 11. The saturated
    # count is kept in the output so the reader sees what was set aside.
    counts["statistical_if_applied"] = len(
        [n for n in alive if not data["statistical"].get(n, {}).get("rejected", False)])

    alive = [n for n in alive if n in data["stressed"]]
    counts["stress"] = len(alive)

    alive = [n for n in alive if n not in data["near_cash"]]
    counts["deployed"] = len(alive)
    return counts


def main() -> int:
    configure_logging()
    views: dict[str, list[dict[str, Any]]] = {"equal_token": [], "equal_n": []}

    for view, limit in (("equal_token", None), ("equal_n", EQUAL_N)):
        _log.info("--- %s view%s ---", view, "" if limit is None else f", first {limit} draws")
        header = f"{'arm':4}" + "".join(f"{s:>11}" for s in STAGES) + f"{SATURATED:>24}"
        _log.info(header)
        for arm in ARMS:
            data = _load(arm)
            names = data["order"] if limit is None else data["order"][:limit]
            counts = _funnel(data, names)
            survival = counts["deployed"] / counts["generated"] if counts["generated"] else 0.0
            views[view].append({
                "arm": arm, "paradigm": ARMS[arm],
                "counts": counts,
                "full_stack_survival": survival,
            })
            row = (f"{arm:<4}" + "".join(f"{counts[s]:>11}" for s in STAGES)
                   + f"{counts[SATURATED]:>24}")
            _log.info("%s   %5.2f%%", row, survival * 100)

    OUT.write_text(json.dumps({
        "stages": list(STAGES),
        "equal_n_size": EQUAL_N,
        "note": ("equal_token is the primary design; equal_n is secondary and exploratory, and "
                 "discards the budget advantage the expensive arms were given."),
        "views": views,
    }, indent=2, sort_keys=True), encoding="utf-8")
    _log.info("written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
