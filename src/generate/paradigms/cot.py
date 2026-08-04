"""G2 — chain of thought: one call, reasoning elicited before the code.

The cheapest possible scaffolding, and the one most likely to be confounded with compute. A single
call that reasons first generates more tokens than a single call that does not, so G2 is already
a different budget point than G1 even though both make one call. The token accountant is what
makes that visible; call counts alone would hide it.

The task specification is P1's, unmodified. Only an instruction about *how to answer* is appended,
which is the intervention under test.
"""

from __future__ import annotations

from typing import Final

from src.audit.semantic import MODEL_TAG, OLLAMA_HOST
from src.generate.paradigms.base import CallRecorder, Draw, attempt_code
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import TokenAccount

#: Appended after P1's specification. It asks for reasoning and it says nothing about leakage,
#: lookahead or point-in-time discipline — warning a generator away from the failure modes being
#: measured would describe the prompt rather than the model. That is P1's reasoning, carried over.
COT_SUFFIX: Final[str] = """

Before writing any code, think the problem through in a section headed REASONING. Work through,
in order: what economic mechanism could make this theme produce returns, what would have to be
true of the data for that mechanism to be measurable, how the signal should be computed from the
accessors available to you, and what would make the strategy lose money.

Then write the finished strategy as a single Python code block fenced with ```python. Put nothing
after the code block.
"""


class ChainOfThought:
    """One call that reasons in the open before emitting the strategy."""

    name = "G2_cot"

    def draw(
        self,
        index: int,
        *,
        base_seed: int = 42,
        model_tag: str = MODEL_TAG,
        host: str = OLLAMA_HOST,
        account: TokenAccount | None = None,
    ) -> Draw:
        theme = theme_for(index)
        recorder = CallRecorder(
            self.name, index, base_seed=base_seed, model_tag=model_tag, host=host, account=account
        )
        outcomes: list[str] = []
        prompt = build_prompt(theme) + COT_SUFFIX
        source = attempt_code(recorder, prompt, stage=0, outcomes=outcomes)
        return recorder.finish(theme, source, outcomes)
