"""How much of an arm's zero redundancy is diversity, and how much is a small sample.

**Exploratory. Not pre-registered.** Added after Checkpoint 4.1 at the PI's direction, whose own
answer to Q1 declined to read G6's 0.0% redundancy as evidence for their pre-registered
graph-of-thoughts prediction: "Twelve traded strategies are simply too few to distinguish 'genuinely
diverse' from 'we didn't sample enough to see a duplicate.'" This puts a number on that. Every
result it produces must be labelled exploratory in the same sentence as the result, per RULE 10.

**The question.** R(G) is the fraction of traded output inside an exact-duplicate cluster. The
equal-token design of Amendment 2 gives the arms very unequal traded counts -- 115 for G1 against 12
for G6 -- and duplicates are counted over *pairs*, which grow quadratically. G6 has 66 pairs to G1's
6,555, so it has roughly one per cent of G1's opportunity to show a duplicate at all.

**The model, and what it assumes.** Take the duplicate rate per pair from G1 and G2 pooled, the two
arms with enough pairs to estimate it, and ask how often an arm of each size would show *no*
duplicate at that same rate. A high probability means the arm's zero is uninformative about its
diversity.

Pairs are treated as independent, which they are not: duplicates arrive in clusters, so a cluster of
three contributes three correlated pairs. That inflates the effective sample and makes the reported
probabilities **conservative in the direction of finding the zero informative** -- the true
probability of seeing no duplicate is if anything higher than stated. Stated here rather than
discovered by a referee.

Usage:
    python scripts/paradigm_redundancy_power.py
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

REDUNDANCY = Path("benchmarks/generationbench/redundancy.json")
OUT = Path("benchmarks/generationbench/redundancy_power.json")
# The arms with enough pairs to estimate a rate at all. Chosen by pair count, not by their results.
REFERENCE = ("G1", "G2")


def duplicate_pairs(arm: dict[str, object]) -> int:
    """Pairs of candidates sharing a cluster, summed over clusters."""
    clusters = arm["clusters"]
    assert isinstance(clusters, list)
    return sum(comb(len(cluster), 2) for cluster in clusters)


def main() -> int:
    configure_logging()
    arms = json.loads(REDUNDANCY.read_text(encoding="utf-8"))["arms"]

    dup = sum(duplicate_pairs(arms[a]) for a in REFERENCE)
    pairs = sum(comb(int(arms[a]["n_traded"]), 2) for a in REFERENCE)
    rate = dup / pairs
    _log.info("reference duplicate rate from %s: %d/%d pairs = %.6f (1 in %.0f)",
              "+".join(REFERENCE), dup, pairs, rate, 1 / rate)

    records = []
    for name, arm in arms.items():
        n = int(arm["n_traded"])
        n_pairs = comb(n, 2)
        p_zero = (1 - rate) ** n_pairs
        records.append({
            "arm": name,
            "paradigm": arm["paradigm"],
            "n_traded": n,
            "n_pairs": n_pairs,
            "duplicate_pairs": duplicate_pairs(arm),
            "redundancy_of_traded": arm["redundancy_of_traded"],
            "p_zero_duplicates_at_reference_rate": p_zero,
        })
        _log.info("%-3s %3d traded, %5d pairs, %2d duplicate; R=%.1f%%; "
                  "P(zero at reference rate) = %.1f%%", name, n, n_pairs,
                  duplicate_pairs(arm), arm["redundancy_of_traded"] * 100, p_zero * 100)

    OUT.write_text(json.dumps({
        "status": "EXPLORATORY -- not pre-registered; added after Checkpoint 4.1",
        "reference_arms": list(REFERENCE),
        "reference_duplicate_pairs": dup,
        "reference_total_pairs": pairs,
        "reference_rate_per_pair": rate,
        "independence_caveat": (
            "Pairs are treated as independent though duplicates arrive in clusters. The reported "
            "P(zero) is therefore conservative: the true probability of observing no duplicate is "
            "if anything higher."
        ),
        "arms": records,
    }, indent=2, sort_keys=True), encoding="utf-8")
    _log.info("written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
