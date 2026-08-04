"""G4 — planning: a plan, a signal design, a risk design, then an implementation.

Four calls in a fixed pipeline, each seeing the output of the last. The hypothesis under test is
that decomposing the task produces better strategies than asking for one in a single breath. The
competing hypothesis, which RQ3 states in advance, is that it mostly produces longer prompts and
narrower output.

Only the final stage sees P1's specification in full. The earlier stages are given the theme and
the interface constraints they need and nothing else, because a planner handed the complete code
contract tends to write code instead of a plan.
"""

from __future__ import annotations

from typing import Final

from src.audit.semantic import MODEL_TAG, OLLAMA_HOST
from src.generate.paradigms.base import CallRecorder, Draw, attempt_code
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import TokenAccount

PLAN_PROMPT: Final[str] = """You are planning a systematic equity trading strategy for the Indian
market, on the theme: {theme}.

Do not write code. In at most 200 words, state the plan: the economic mechanism you intend to
exploit, why it might persist, and the shape of the rule that would exploit it.
"""

SIGNAL_PROMPT: Final[str] = """Here is a plan for a systematic equity strategy:

{plan}

Do not write code. In at most 200 words, specify the signal precisely: what is computed from daily
open, high, low, close and volume, over what lookback, and how candidates are ranked against each
other on a given day.
"""

RISK_PROMPT: Final[str] = """Here is a plan and a signal specification for a systematic equity
strategy:

{plan}

{signal}

Do not write code. In at most 150 words, specify position sizing and risk limits: how many names
are held, how they are weighted, and what bounds the loss.
"""

IMPLEMENT_SUFFIX: Final[str] = """

Implement exactly this design, which has already been agreed:

PLAN
{plan}

SIGNAL
{signal}

RISK
{risk}

Write the finished strategy as a single Python code block fenced with ```python. Put nothing after
the code block.
"""


class Planning:
    """Planner, then signal designer, then risk designer, then implementer."""

    name = "G4_planning"

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
        plan = recorder.ask(PLAN_PROMPT.format(theme=theme), stage=1)
        signal = recorder.ask(SIGNAL_PROMPT.format(plan=plan), stage=2)
        risk = recorder.ask(RISK_PROMPT.format(plan=plan, signal=signal), stage=3)

        outcomes: list[str] = []
        prompt = build_prompt(theme) + IMPLEMENT_SUFFIX.format(plan=plan, signal=signal, risk=risk)
        source = attempt_code(recorder, prompt, stage=4, outcomes=outcomes)
        return recorder.finish(theme, source, outcomes)
