"""G5 — reflection: generate, critique the draft, rewrite it.

Three calls. The critique stage is the intervention, and it is the one place in P4 where a
paradigm could accidentally be handed the auditor's job. It is not: the critic is asked to review
the strategy as a piece of engineering and economics, using the same information the generator
had. It is never shown an audit verdict, a backtest result, or any output of the frozen stack.

**Why that restriction is load-bearing.** If the critic were told what the static layer flags, this
arm's audit pass rate would measure how well the prompt describes the auditor, not how well
reflection works. The whole comparison would collapse into a tautology, and it would do so
invisibly, in a direction that flatters the paradigm.
"""

from __future__ import annotations

from typing import Final

from src.audit.semantic import MODEL_TAG, OLLAMA_HOST
from src.generate.paradigms.base import CallRecorder, Draw, attempt_code
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import TokenAccount

CRITIQUE_PROMPT: Final[str] = """Here is a draft systematic equity trading strategy:

```python
{draft}
```

Do not rewrite it. In at most 250 words, criticise it: where the stated rationale and the code
disagree, where the rule would fail, what is fragile about the parameters, and anything that would
stop it running. Be specific and cite lines. If it is sound, say which parts and why.
"""

REWRITE_SUFFIX: Final[str] = """

Here is your earlier draft:

```python
{draft}
```

Here is a critique of it:

{critique}

Rewrite the strategy addressing the critique. Keep what the critique found sound. Write the result
as a single Python code block fenced with ```python. Put nothing after the code block.
"""


class Reflection:
    """Draft, critique, rewrite — with the critic blind to the frozen stack."""

    name = "G5_reflection"

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

        # The draft is not required to conform: a broken draft is exactly what the critic exists to
        # catch, and rejecting it here would hide the paradigm's whole mechanism of action.
        draft = attempt_code(recorder, build_prompt(theme), stage=0, outcomes=outcomes)
        critique = recorder.ask(CRITIQUE_PROMPT.format(draft=draft), stage=5)

        prompt = build_prompt(theme) + REWRITE_SUFFIX.format(draft=draft, critique=critique)
        source = attempt_code(recorder, prompt, stage=6, outcomes=outcomes)
        return recorder.finish(theme, source, outcomes)
