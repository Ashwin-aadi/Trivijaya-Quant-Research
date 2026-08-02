"""Plain-English stress narratives: the explainability layer over the fragility measurements.

A risk committee is not handed ``F = 0.618``. It is handed a sentence: *this strategy holds a
concentrated book for five weeks at a time and loses most of its edge when volatility regime 2
arrives*. This module asks a local model to write that sentence — and constrains it so the sentence
can be checked.

Three constraints, because an unconstrained language model will produce a fluent, confident, and
unverifiable story every single time:

1. **Only measured facts go into the prompt.** Regime Sharpe ratios with their sample sizes,
   fragility, turnover, holding period, concentration. No price history, no narrative context, no
   invitation to speculate about what "the market" was doing.
2. **The facts are stored beside the narrative** in the output file, so the PI reading a narrative
   at Checkpoint 2.2 can check every claim in it against the numbers it was given, and catch the
   model asserting something it was never told.
3. **The regimes are numbered, not named.** The Phase 2.0 labels are ``0..3`` from an HMM and carry
   no economic interpretation. Telling the model "regime 2 is a crisis" would put the conclusion in
   the prompt and get it back as a finding.

Determinism follows :mod:`src.audit.semantic`: temperature 0, seed 42, the same model tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.audit.semantic import (
    MODEL_TAG,
    OLLAMA_HOST,
    REQUEST_TIMEOUT_SECONDS,
    _post_generate,
)

#: What the model is asked to produce. Two or three sentences: long enough to name a mechanism,
#: short enough that every clause can be checked against the facts table beside it.
INSTRUCTION = """You are a risk analyst writing one entry in a strategy risk register.

Below are measured statistics for one trading strategy on Indian equities, 2020-2024. The market
regimes are numbered labels from a hidden Markov model fitted on realised volatility. They have no
names and no economic interpretation was supplied to you.

Write 2-3 sentences describing when this strategy is most and least reliable, and what property of
the strategy explains it. Rules:
- Use only the numbers given. Do not invent events, dates, sectors, or macroeconomic causes.
- Do not claim to know what a regime number means. Refer to them as "regime 0", "regime 1", etc.
- If the statistics do not support a clear conclusion, say that instead of manufacturing one.
- Do not restate the numbers; explain what they imply.

Return JSON with exactly two keys: "narrative" (string) and "confidence" (0.0 to 1.0).
"""


@dataclass(frozen=True)
class StressNarrative:
    """One generated narrative with the exact facts it was given, so it can be audited."""

    name: str
    narrative: str
    confidence: float
    facts: dict[str, Any]
    model_tag: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "narrative": self.narrative,
            "confidence": self.confidence,
            "facts": self.facts,
            "model_tag": self.model_tag,
        }


def build_facts(
    name: str,
    fragility: dict[str, Any],
    characteristics: dict[str, Any],
) -> dict[str, Any]:
    """The measured facts about one strategy, and nothing else.

    Regime Sharpe ratios carry their session counts, so a narrative that leans on a regime the
    strategy barely traded in can be recognised as leaning on a small sample.
    """
    return {
        "strategy": name,
        "fragility_across_regimes": fragility.get("fragility_across_regimes"),
        "fragility_across_paths": fragility.get("fragility_across_paths"),
        "mean_regime_sharpe": fragility.get("mean_regime_sharpe"),
        "sharpe_by_regime": fragility.get("regime_sharpe", {}),
        "sessions_by_regime": fragility.get("regime_sessions", {}),
        "mean_turnover_per_session": characteristics.get("mean_turnover"),
        "mean_holding_period_sessions": characteristics.get("mean_holding_period"),
        "effective_number_of_holdings": characteristics.get("effective_holdings"),
        "book_similarity_after_21_sessions": characteristics.get("book_similarity_21d"),
        "share_of_sessions_holding_nothing": characteristics.get("cash_session_rate"),
    }


def build_prompt(facts: dict[str, Any]) -> str:
    """Instruction plus the facts as a readable block. No examples, deliberately.

    A worked example in the prompt would supply a mechanism the model could copy, and the resulting
    narrative would then be evidence about the example rather than about the strategy.
    """
    lines = [f"  {key}: {value}" for key, value in facts.items()]
    return f"{INSTRUCTION}\nMEASURED STATISTICS\n" + "\n".join(lines) + "\n"


def generate(
    name: str,
    facts: dict[str, Any],
    *,
    model_tag: str = MODEL_TAG,
    host: str = OLLAMA_HOST,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> StressNarrative:
    """One narrative. Raises :class:`SemanticAuditUnavailable` if the server is not reachable."""
    raw = _post_generate(build_prompt(facts), model_tag=model_tag, host=host, timeout=timeout)
    return _parse(name, raw, facts, model_tag)


def _parse(name: str, raw: str, facts: dict[str, Any], model_tag: str) -> StressNarrative:
    """Read the model's JSON. A malformed reply is kept as the raw text, never silently dropped.

    A narrative the model failed to format is still evidence about the model, and discarding it
    would quietly improve the apparent quality of the layer being evaluated.
    """
    import json

    from src.audit.semantic import _extract_json_object

    try:
        parsed = json.loads(_extract_json_object(raw))
        narrative = str(parsed.get("narrative", "")).strip()
        confidence = float(parsed.get("confidence", float("nan")))
    except (ValueError, TypeError, AttributeError):
        return StressNarrative(
            name=name,
            narrative=f"[unparsed model output] {raw.strip()[:600]}",
            confidence=float("nan"),
            facts=facts,
            model_tag=model_tag,
        )
    return StressNarrative(
        name=name,
        narrative=narrative or f"[empty narrative] {raw.strip()[:300]}",
        confidence=confidence,
        facts=facts,
        model_tag=model_tag,
    )
