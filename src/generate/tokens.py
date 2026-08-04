"""Count the tokens a generation paradigm spends, so paradigms can be compared at equal budget.

RULE 11 makes an unequal-budget comparison invalid, and a budget is measured in generated tokens.
Nothing else in this repository measures them: :func:`src.generate.generator._post` issued every P1
call and read only the ``response`` field, discarding the ``eval_count`` Ollama returns beside it.
A paradigm that makes five calls per strategy is not better than one that makes a single call, and
without this module there is no way to say so.

**Output tokens are the budget.** ``eval_count`` is what the model generated; ``prompt_eval_count``
is what it read. Both are recorded, because a paradigm that stuffs a long plan back into the next
prompt shifts cost from generation into context and would otherwise look free, but the budget that
RULE 11 matches on is the generated one.

**A failed call still spent its tokens.** Usage is accumulated from the response, so a draw that
produced unparseable code is charged for what it produced. Charging only successes would make the
noisiest paradigm look cheapest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Usage:
    """Tokens consumed by one call to the model."""

    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @classmethod
    def from_ollama(cls, body: dict[str, Any]) -> Usage:
        """Read the counts out of an Ollama ``/api/generate`` response body.

        Missing keys become zero rather than raising. Ollama omits them on some error paths, and a
        crash here would lose an otherwise usable draw; a zero is visibly wrong in the accounts,
        which is the failure mode we can detect later.
        """
        return cls(
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            output_tokens=int(body.get("eval_count", 0) or 0),
        )


@dataclass
class TokenAccount:
    """Running totals per paradigm, and the per-strategy cost that RULE 11 matches on."""

    #: Summed usage, keyed by paradigm name.
    usage: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    #: Model calls issued, keyed by paradigm name. A paradigm's call count is not its token
    #: count — a five-call paradigm whose calls are short may still be cheap.
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    #: Draws attempted, whether or not they yielded a usable strategy.
    draws: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    #: Draws that produced a conforming strategy.
    accepted: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_call(self, paradigm: str, usage: Usage) -> None:
        self.usage[paradigm] = self.usage[paradigm] + usage
        self.calls[paradigm] += 1

    def record_draw(self, paradigm: str, *, accepted: bool) -> None:
        self.draws[paradigm] += 1
        if accepted:
            self.accepted[paradigm] += 1

    def output_tokens_per_accepted(self, paradigm: str) -> float | None:
        """Generated tokens spent per usable strategy, or None if nothing was accepted.

        This is the figure a compute-matched control is built against: it says what one usable
        strategy costs, which is the unit a researcher actually buys. None rather than infinity so
        that a paradigm which accepted nothing cannot be silently averaged into a comparison.
        """
        accepted = self.accepted[paradigm]
        if accepted == 0:
            return None
        return self.usage[paradigm].output_tokens / accepted

    def to_dict(self) -> dict[str, dict[str, float | int | None]]:
        """A JSON-serialisable summary, for the run manifest."""
        return {
            name: {
                "prompt_tokens": self.usage[name].prompt_tokens,
                "output_tokens": self.usage[name].output_tokens,
                "calls": self.calls[name],
                "draws": self.draws[name],
                "accepted": self.accepted[name],
                "output_tokens_per_accepted": self.output_tokens_per_accepted(name),
            }
            for name in sorted(set(self.usage) | set(self.draws))
        }
