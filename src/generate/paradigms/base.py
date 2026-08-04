"""What every generation paradigm shares: the call, the seed scheme, and the acceptance rule.

A paradigm is a procedure, not a prompt. It may issue one model call or nine, in any arrangement,
and it must report what that cost. Everything a paradigm is allowed to vary lives in its own
module; everything that must be identical across arms lives here, because a difference that leaks
into the shared machinery would be indistinguishable from an effect of the paradigm.

**Three things are deliberately not a paradigm's to choose.**

*The task specification.* Every arm builds its prompt from :func:`src.generate.prompts.build_prompt`
on the theme :func:`src.generate.prompts.theme_for` assigns to its index. The digest is P1's, so
P4's corpus is directly comparable to P1's in a way the generator-validation addendum's was not.

*The acceptance rule.* A draw is usable exactly when P1's conformance check says so. The private
helpers are imported from :mod:`src.generate.generator` rather than reimplemented: an arm judged by
a subtly different rule would produce a yield difference that is an artefact of the harness.

*The auditor.* No paradigm may consult any layer of the frozen stack while generating. A paradigm
that selects on the instrument measuring it has decided its own result. :class:`Draw` carries no
audit field for that reason, and the evolutionary arm's fitness function is documented as
backtest-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol

from src.common.log import get_logger
from src.generate.generator import (
    MAX_ATTEMPTS,
    _class_name,
    _conformance_failure,
    _post,
    extract_code,
)
from src.generate.tokens import TokenAccount, Usage

_log = get_logger(__name__)

#: Seed spacing between stages of one paradigm. Wide enough that no stage of any draw can collide
#: with another stage of any other draw for index counts far beyond anything P4 will generate.
STAGE_STRIDE: Final[int] = 100_000

#: Seed spacing between retries within a stage, matching P1's scheme exactly.
ATTEMPT_STRIDE: Final[int] = 10_000


class ParadigmError(Exception):
    """A paradigm was configured in a way that would make its arm incomparable."""


def seed_for(base_seed: int, index: int, stage: int, attempt: int) -> int:
    """The seed for one call, unique across draws, stages and retries.

    At ``stage=0`` this reduces to P1's ``base_seed + index + attempt * 10_000``, so the plain
    single-call paradigm reproduces the reference corpus bit for bit rather than merely resembling
    it. That is what lets P1's 1,550 draws serve as P4's control arm.
    """
    return base_seed + index + stage * STAGE_STRIDE + attempt * ATTEMPT_STRIDE


@dataclass(frozen=True)
class Draw:
    """One paradigm's complete attempt at one strategy, and everything it cost to make."""

    index: int
    paradigm: str
    theme: str
    source: str
    class_name: str
    #: "evaluated" when the source conforms; otherwise why it does not. The vocabulary is P1's, so
    #: outcomes are poolable with the reference corpus.
    outcome: str
    usage: Usage
    #: Model calls issued for this draw, retries included.
    calls: int
    #: Distinct strategy candidates this draw produced and evaluated internally, retries and
    #: discarded intermediates included. **The trial ledger increments by this, not by one.** For
    #: the single-shot arms it is the number of code attempts; for the evolutionary arm it is the
    #: whole population across every generation, most of which is thrown away. That asymmetry is
    #: RQ4 — whether honest trial accounting erases the gains of iterating — and it is a
    #: measurement, not an accounting inconvenience to be smoothed over.
    candidates_evaluated: int = 1
    attempt_outcomes: tuple[str, ...] = ()
    #: Intermediate texts — plans, critiques, agent turns — in the order they were produced. Kept
    #: so a reader can see what the scaffolding actually did rather than trusting that it helped.
    trace: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        return self.outcome == "evaluated"


class Paradigm(Protocol):
    """A procedure mapping a draw index to a strategy, reporting its own token cost."""

    name: str

    def draw(
        self,
        index: int,
        *,
        base_seed: int = 42,
        model_tag: str = ...,
        host: str = ...,
        account: TokenAccount | None = None,
    ) -> Draw:
        """Produce draw ``index``, recording usage into ``account`` if one is given."""
        ...


class CallRecorder:
    """Issues model calls for one draw and accumulates what they cost.

    Exists so a paradigm's own code reads as the procedure it implements — plan, then design, then
    implement — without every module repeating the seeding and accounting. A paradigm that forgot
    to accumulate would look cheap, and cheapness is the primary axis of comparison.
    """

    def __init__(
        self,
        paradigm: str,
        index: int,
        *,
        base_seed: int,
        model_tag: str,
        host: str,
        account: TokenAccount | None = None,
    ) -> None:
        self.paradigm = paradigm
        self.index = index
        self.base_seed = base_seed
        self.model_tag = model_tag
        self.host = host
        self.account = account
        self.usage = Usage()
        self.calls = 0
        self.trace: list[str] = []

    def ask(self, prompt: str, *, stage: int, attempt: int = 1, keep: bool = True) -> str:
        """One model call. ``keep`` records the reply in the trace; code stages set it False."""
        seed = seed_for(self.base_seed, self.index, stage, attempt)
        text, usage = _post(prompt, seed, model_tag=self.model_tag, host=self.host)
        self.usage = self.usage + usage
        self.calls += 1
        if self.account is not None:
            self.account.record_call(self.paradigm, usage)
        if keep:
            self.trace.append(text)
        return text

    def finish(
        self,
        theme: str,
        source: str,
        outcomes: list[str],
        *,
        candidates_evaluated: int | None = None,
    ) -> Draw:
        """Build the Draw, applying P1's acceptance rule to ``source``.

        ``candidates_evaluated`` defaults to the number of code attempts made, which is right for
        every arm that produces one lineage. An arm that breeds a population must pass its own
        count, or the ledger will undercount it and its deflation will be too generous.
        """
        reason = _conformance_failure(source) if source else "no code produced"
        outcome = "evaluated" if reason is None else _classify(reason)
        if reason is not None:
            _log.warning("%s draw %d unusable: %s", self.paradigm, self.index, reason)
        if self.account is not None:
            self.account.record_draw(self.paradigm, accepted=reason is None)
        return Draw(
            index=self.index,
            paradigm=self.paradigm,
            theme=theme,
            source=source if reason is None else "",
            class_name=_class_name(source) if reason is None else f"Failed{self.index}",
            outcome=outcome,
            usage=self.usage,
            calls=self.calls,
            candidates_evaluated=candidates_evaluated or max(len(outcomes), 1),
            attempt_outcomes=tuple(outcomes),
            trace=tuple(self.trace),
        )


def _classify(reason: str) -> str:
    """Map a conformance failure onto P1's outcome vocabulary."""
    return "syntax_error" if "syntax" in reason else "runtime_error"


def attempt_code(
    recorder: CallRecorder,
    prompt: str,
    *,
    stage: int,
    outcomes: list[str],
) -> str:
    """Ask for code and retry on a non-conforming reply, up to P1's bound.

    Retries are charged as calls, exactly as P1 charges them as trials. A paradigm that quietly
    retried without accounting would buy yield with compute and report it as skill, which is the
    confound RULE 11 exists to close.
    """
    source = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        source = extract_code(recorder.ask(prompt, stage=stage, attempt=attempt, keep=False))
        reason = _conformance_failure(source)
        if reason is None:
            outcomes.append("evaluated")
            return source
        outcomes.append(_classify(reason))
    return source
