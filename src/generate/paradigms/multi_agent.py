"""G6 — multi-agent: an economist, a statistician, a quant and a reviewer, then an implementer.

Five calls, each with a different stated role, the later ones seeing the earlier ones. This is the
most expensive arm per strategy and therefore the one RULE 11 most easily flatters: it makes five
times the calls of plain prompting, so beating plain prompting is unremarkable unless plain
prompting is given the same token budget and allowed to keep its best.

The roles are personas in a prompt, not separate models or separate processes. That is what the
literature this arm represents actually does, and it is stated plainly so nobody reads more
architecture into it than exists.

**No role has access to the frozen stack**, for the reason set out in
:mod:`src.generate.paradigms.reflection`.
"""

from __future__ import annotations

from typing import Final

from src.audit.semantic import MODEL_TAG, OLLAMA_HOST
from src.generate.paradigms.base import CallRecorder, Draw, attempt_code
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import TokenAccount

ECONOMIST_PROMPT: Final[str] = """You are an economist. A systematic equity strategy is to be built
on this theme, for the Indian market: {theme}.

In at most 150 words, and without writing code: what economic mechanism would make this theme
produce returns, who is on the other side of the trade, and why has it not been competed away?
"""

STATISTICIAN_PROMPT: Final[str] = """You are a statistician. An economist proposes this mechanism
for a systematic equity strategy:

{economist}

In at most 150 words, and without writing code: how would you measure this from daily open, high,
low, close and volume? State the estimator, the lookback, and what would make the estimate
unreliable.
"""

QUANT_PROMPT: Final[str] = """You are a quantitative portfolio manager. Here is a proposed
mechanism and its measurement:

{economist}

{statistician}

In at most 150 words, and without writing code: turn this into a trading rule. Entry, exit, how
many names, how weighted, and what bounds the loss.
"""

REVIEWER_PROMPT: Final[str] = """You are a reviewer whose job is to find what is wrong with a
proposal before capital is committed to it. Here is the proposal:

{economist}

{statistician}

{quant}

In at most 150 words, and without writing code: state the strongest objection to this proposal and
what should change because of it. If the proposal is sound, say which part is weakest anyway.
"""

CONSENSUS_SUFFIX: Final[str] = """

A team has already agreed this design. Implement it, taking the reviewer's objection into account:

ECONOMIST
{economist}

STATISTICIAN
{statistician}

PORTFOLIO MANAGER
{quant}

REVIEWER
{reviewer}

Write the finished strategy as a single Python code block fenced with ```python. Put nothing after
the code block.
"""


class MultiAgent:
    """Four role-played analyses, then one implementation of their consensus."""

    name = "G6_multi_agent"

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
        economist = recorder.ask(ECONOMIST_PROMPT.format(theme=theme), stage=1)
        statistician = recorder.ask(
            STATISTICIAN_PROMPT.format(economist=economist), stage=2
        )
        quant = recorder.ask(
            QUANT_PROMPT.format(economist=economist, statistician=statistician), stage=3
        )
        reviewer = recorder.ask(
            REVIEWER_PROMPT.format(
                economist=economist, statistician=statistician, quant=quant
            ),
            stage=4,
        )

        outcomes: list[str] = []
        prompt = build_prompt(theme) + CONSENSUS_SUFFIX.format(
            economist=economist, statistician=statistician, quant=quant, reviewer=reviewer
        )
        source = attempt_code(recorder, prompt, stage=7, outcomes=outcomes)
        return recorder.finish(theme, source, outcomes)
