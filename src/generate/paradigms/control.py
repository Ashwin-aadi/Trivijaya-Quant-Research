"""The compute-matched control: plain prompting given the expensive arm's budget, keeping its best.

RULE 11 states the whole of this module's justification. A paradigm that spends five calls per
strategy is not better than one that spends a single call; it is a different point on a budget
curve. The control is therefore not plain prompting at its natural budget — it is plain prompting
run k times and allowed to keep its best, where k is set by tokens, not by call count.

**k is a token ratio, not a call ratio.** Chain of thought makes one call and still costs more than
plain prompting, because it generates reasoning before the code. Sizing k by calls would give that
arm a free budget increase. :func:`matched_k` divides output tokens by output tokens.

**Draws are resampled as contiguous index blocks, never as a uniform random sample.** P1's corpus
is stratified: ``theme_for(index)`` assigns ``THEMES[index % 12]``, so k consecutive indices cover
k consecutive themes exactly as k fresh draws would. A uniform sample of k indices draws a random
theme mixture with repeats — a different sampling scheme, a different variance, and a confound that
would be invisible in the output. This is the correction that made corpus reuse admissible at all.

**A block containing nothing rankable is a failed control draw, not a discarded one.** Only 225 of
P1's 1,550 candidates executed and took a position, so for small k many blocks contain no usable
strategy. Those blocks yield ``None`` and count in the control's yield denominator. Dropping them
would compare the control's best blocks against every one of the treatment's, which inflates the
control and would be the easiest way to accidentally manufacture a null.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.generate.paradigms.base import ParadigmError


@dataclass(frozen=True)
class ControlDraw:
    """One best-of-k selection from the reference corpus."""

    #: The contiguous corpus indices this draw was allowed to choose among.
    block: tuple[int, ...]
    #: The index selected, or None when nothing in the block was rankable.
    selected: int | None
    #: The selected candidate's score, or None on an empty block.
    score: float | None

    @property
    def usable(self) -> bool:
        return self.selected is not None


def matched_k(treatment_tokens_per_accepted: float, control_tokens_per_accepted: float) -> int:
    """How many plain draws buy one treatment strategy, at equal generated-token budget.

    Rounded **up**. A fractional k has to break somewhere, and rounding up hands the surplus to the
    control — the arm whose advantage we are trying not to overstate. Rounding down would give the
    treatment a budget the control never received, which is precisely the failure RULE 11 names.
    """
    if control_tokens_per_accepted <= 0:
        raise ParadigmError("control token cost must be positive to match a budget against it")
    if treatment_tokens_per_accepted <= 0:
        raise ParadigmError("treatment token cost must be positive")
    return max(1, math.ceil(treatment_tokens_per_accepted / control_tokens_per_accepted))


def resample_blocks(
    corpus_size: int, k: int, draws: int, *, seed: int = 42
) -> list[tuple[int, ...]]:
    """``draws`` contiguous index blocks of length ``k``, drawn with replacement.

    With replacement because the treatment arm's draws are independent and the control's must be
    too; sampling without replacement would give the control a coverage guarantee the treatment
    never had.
    """
    if k < 1:
        raise ParadigmError("k must be at least 1")
    if corpus_size < k:
        raise ParadigmError(f"corpus of {corpus_size} cannot supply a block of {k}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, corpus_size - k + 1, size=draws)
    return [tuple(range(int(start), int(start) + k)) for start in starts]


def best_of_block(block: Sequence[int], scores: Mapping[int, float | None]) -> ControlDraw:
    """Pick the highest-scoring rankable index in ``block``.

    Ties break to the lowest index, which is deterministic and independent of the scores, so a
    corpus with many identical strategies — and P1's has 11 such clusters — cannot have its
    selection quietly decided by dictionary ordering.
    """
    ranked = [(i, scores.get(i)) for i in block]
    usable = [(i, s) for i, s in ranked if s is not None]
    if not usable:
        return ControlDraw(block=tuple(block), selected=None, score=None)
    best_index, best_score = min(usable, key=lambda pair: (-pair[1], pair[0]))
    return ControlDraw(block=tuple(block), selected=best_index, score=best_score)


def run_control(
    corpus_size: int,
    scores: Mapping[int, float | None],
    *,
    k: int,
    draws: int,
    seed: int = 42,
) -> list[ControlDraw]:
    """The full control arm: ``draws`` best-of-k selections over contiguous blocks."""
    return [
        best_of_block(block, scores)
        for block in resample_blocks(corpus_size, k, draws, seed=seed)
    ]
