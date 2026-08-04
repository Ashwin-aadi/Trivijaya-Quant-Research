"""G1 — plain prompting: one call, P1's prompt, nothing added.

The reference procedure and the baseline every other arm is measured against. It is deliberately
the cheapest thing anyone would actually do, because RULE 11's control is this method run enough
times to spend the expensive method's budget — not this method at its natural budget.

**This arm is also the calibration the control needs.** P1's 1,550 draws are reused as the
compute-matched control, but they were generated before anything counted tokens, so their recorded
usage is zero. Running this paradigm live is the only way to learn what one plain draw costs, and
without that figure best-of-k cannot be sized and RULE 11 cannot be enforced.
"""

from __future__ import annotations

from src.audit.semantic import MODEL_TAG, OLLAMA_HOST
from src.generate.paradigms.base import CallRecorder, Draw, attempt_code
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import TokenAccount


class PlainPrompting:
    """One call per strategy, at stage 0 — bit-identical to P1's generator."""

    name = "G1_plain"

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
        source = attempt_code(recorder, build_prompt(theme), stage=0, outcomes=outcomes)
        return recorder.finish(theme, source, outcomes)
