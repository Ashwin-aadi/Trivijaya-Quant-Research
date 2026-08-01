"""Guard the sign of every auditor layer's contribution to the abstention ranking.

This file exists because that sign was wrong, and nothing caught it. ``soundness`` negated each
layer's ``confidence`` on the assumption it meant "confidence that something is wrong". The static
layer stores a finding count there and the statistical layer stores a constant 1.0 on records it
always rejects, so for both the assumption held. The semantic layer stores confidence *in its
label* - about 0.95 for a confident "consistent", about 0.85 for a confident rejection - so
negating it ranked the strategies that layer had rejected **above** the ones it had cleared. Four of
the seven ablation configurations were ranked backwards, including the one the paper quoted as its
best holdout AUAP.

The bug was invisible to every existing test: each layer's own tests check its labels, which were
correct, and the abstention tests check the curve given a ranking, which was computed correctly from
a ranking that meant the opposite of what it claimed. Only the join between them was wrong. So the
tests below assert exactly that join, in the direction a reader can check by eye:

* an accepted candidate must outrank a rejected one, for every layer, at every confidence;
* the correction must leave the static and statistical layers numerically untouched, which is what
  makes it legitimate to leave their published results standing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.run_ablation import LAYERS, combined, soundness

from src.eval.abstention import abstention_curve

CORPUS_AUDIT = Path("runs/pooled/audit_results.json")


def _audit(**layers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return dict(layers)


# --- the inversion itself -------------------------------------------------------


@pytest.mark.parametrize("layer", LAYERS)
def test_an_acceptance_outranks_a_rejection(layer: str) -> None:
    """The whole bug in one assertion, stated per layer so a future layer inherits the check."""
    audit = _audit(**{layer: {
        "cleared": {"rejected": False, "confidence": 0.95},
        "flagged": {"rejected": True, "confidence": 0.85},
    }})
    assert soundness(audit, layer, "cleared") > soundness(audit, layer, "flagged")


def test_semantic_confidence_does_not_invert_the_order() -> None:
    """A rejection held with *less* confidence than an acceptance still ranks below it.

    This is the specific numeric shape of the corpus - 0.95 for "consistent", 0.85 for a defect -
    and it is what made the old code look plausible: negation turned the smaller number into the
    larger score.
    """
    audit = _audit(semantic={
        "consistent_095": {"rejected": False, "label": "consistent", "confidence": 0.95},
        "mismatch_085": {"rejected": True, "label": "rationale_implementation_mismatch",
                         "confidence": 0.85},
        "mismatch_095": {"rejected": True, "label": "rationale_implementation_mismatch",
                         "confidence": 0.95},
    })
    scores = {n: soundness(audit, "semantic", n) for n in audit["semantic"]}
    assert scores["consistent_095"] > scores["mismatch_085"] > scores["mismatch_095"]


def test_more_confidence_in_a_rejection_ranks_lower() -> None:
    audit = _audit(semantic={
        "unsure": {"rejected": True, "confidence": 0.6},
        "certain": {"rejected": True, "confidence": 0.99},
    })
    assert soundness(audit, "semantic", "unsure") > soundness(audit, "semantic", "certain")


def test_an_unscored_candidate_sits_between_acceptance_and_rejection() -> None:
    """Absence of evidence must not be evidence of soundness, nor of a defect."""
    audit = _audit(semantic={
        "cleared": {"rejected": False, "confidence": 0.95},
        "flagged": {"rejected": True, "confidence": 0.85},
    })
    unscored = soundness(audit, "semantic", "never_audited")
    assert soundness(audit, "semantic", "flagged") < unscored < soundness(audit, "semantic",
                                                                         "cleared")


def test_a_record_without_a_rejected_field_is_treated_as_a_rejection() -> None:
    """Fail toward flagging. A layer that reported a confidence but no verdict is not a pass."""
    audit = _audit(semantic={"partial": {"confidence": 0.9}})
    assert soundness(audit, "semantic", "partial") == pytest.approx(-0.9)


# --- the ranking that is actually consumed --------------------------------------


def test_the_curve_keeps_cleared_candidates_first() -> None:
    """End to end: the same defect, expressed where it did its damage - which set survives at 5%.

    Cleared candidates are given the *worse* performance here on purpose. If the ranking were still
    inverted the curve would look better at low coverage, so a test that gave the cleared set the
    better numbers could pass under the bug.
    """
    audit = _audit(semantic={
        f"cleared_{i}": {"rejected": False, "confidence": 0.95} for i in range(10)
    } | {
        f"flagged_{i}": {"rejected": True, "confidence": 0.85} for i in range(10)
    })
    names = sorted(audit["semantic"])
    performance = {n: (2.0 if n.startswith("flagged") else -1.0) for n in names}

    curve = abstention_curve(combined(audit, ("semantic",), names), performance,
                             coverages=(0.5, 1.0))
    assert curve.performance[0] == pytest.approx(-1.0), (
        "at 50% coverage the retained set must be the ten cleared candidates"
    )


# --- the correction is confined to the semantic layer ---------------------------


def _old_soundness(audit: dict[str, Any], layer: str, name: str) -> float:
    """The pre-v1.1 formula, kept verbatim so 'unchanged' is demonstrated rather than asserted."""
    record = audit.get(layer, {}).get(name)
    if not record:
        return 0.0
    return -float(record.get("confidence", 0.0))


@pytest.mark.skipif(not CORPUS_AUDIT.exists(), reason="requires the frozen corpus audit results")
@pytest.mark.parametrize("layer", ["static", "statistical"])
def test_the_correction_leaves_the_other_layers_bit_identical(layer: str) -> None:
    """Why the static, statistical and static+statistical results were not rerun.

    On this corpus the static layer records ``rejected`` exactly when its finding count is non-zero,
    so a cleared record scores +0.0 and a flagged one -n either way; the statistical layer rejects
    every record it scores, so every value is -1.0 either way. Both are properties of the frozen
    data, not of the code, which is why they are checked against the data.
    """
    audit = json.loads(CORPUS_AUDIT.read_text(encoding="utf-8"))
    for name in audit[layer]:
        assert soundness(audit, layer, name) == _old_soundness(audit, layer, name), name


@pytest.mark.skipif(not CORPUS_AUDIT.exists(), reason="requires the frozen corpus audit results")
def test_the_semantic_layer_did_change_on_the_corpus() -> None:
    """The counterpart: if nothing changed there was no bug, and the rerun was unjustified."""
    audit = json.loads(CORPUS_AUDIT.read_text(encoding="utf-8"))
    changed = [
        name for name in audit["semantic"]
        if soundness(audit, "semantic", name) != _old_soundness(audit, "semantic", name)
    ]
    assert len(changed) == 1396, (
        f"expected the 1,396 semantically cleared candidates to change sign, got {len(changed)}"
    )
