"""G6 — graph of thoughts: independent proposals, aggregated two ways, merged, then implemented.

Replaces the multi-agent arm by PI ruling of 2026-08-04, before any strategy was generated. The
retired module is kept at :mod:`src.generate.paradigms.multi_agent` and is not run.

**What makes this a graph rather than a chain.** The multi-agent arm was a chain: each role saw the
previous role's output, so the fourth call could only elaborate the first. Here the three layer-one
proposals answer the *same* question independently and never see each other, and the operation that
defines the paradigm is **aggregation** — combining independent thoughts into one better thought
(Besta et al., *Graph of Thoughts: Solving Elaborate Problems with Large Language Models*, 2023).
Two aggregations are made under different criteria and are themselves merged, so information flows
through a directed acyclic graph with in-degree greater than one at three nodes.

Seven calls: three proposals, two aggregations, one merge, one implementation. Six of the seven are
word-capped, so this arm's token cost is well below seven times plain prompting's and the token
accountant measures how far below rather than assuming it.

**No node has access to the frozen stack**, for the reason set out in
:mod:`src.generate.paradigms.reflection`. Aggregation is over the model's own proposals only.
"""

from __future__ import annotations

from typing import Final

from src.audit.semantic import MODEL_TAG, OLLAMA_HOST
from src.generate.paradigms.base import CallRecorder, Draw, attempt_code
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import TokenAccount

PROPOSAL_PROMPT: Final[str] = """A systematic equity strategy is to be built on this theme, for the
Indian market: {theme}.

In at most 200 words, and without writing code, propose one complete design: the economic mechanism,
how it is measured from daily open, high, low, close and volume, the entry and exit rule, how many
names are held and how they are weighted, and what bounds the loss.

Commit to one design. Do not list alternatives.
"""

#: Aggregation A — keep what the proposals agree on. The conservative reading of a set of
#: independent thoughts.
AGGREGATE_AGREEMENT_PROMPT: Final[str] = """Three independent designs were proposed for the same
systematic equity strategy.

DESIGN 1
{a}

DESIGN 2
{b}

DESIGN 3
{c}

In at most 200 words, and without writing code: state the single design that keeps only what these
three agree on. Where they disagree, choose the simplest option and say why. The result must be one
complete design, not a comparison.
"""

#: Aggregation B — keep the strongest single element of each, whether or not the others share it.
#: A different aggregation criterion over the same inputs, which is the operation the chain-shaped
#: paradigms cannot express.
AGGREGATE_STRENGTH_PROMPT: Final[str] = """Three independent designs were proposed for the same
systematic equity strategy.

DESIGN 1
{a}

DESIGN 2
{b}

DESIGN 3
{c}

In at most 200 words, and without writing code: take the strongest single element from each — one
idea from each design, whether or not the other two share it — and combine them into one coherent
design. State which element came from which design and why it is the strongest.
"""

MERGE_PROMPT: Final[str] = """Two designs were derived from the same three proposals under different
criteria. The first kept what they agreed on. The second combined the strongest element of each.

CONSERVATIVE
{agreement}

AMBITIOUS
{strength}

In at most 200 words, and without writing code: produce the single design that will actually be
implemented. Where the two disagree, decide, and give the reason for the decision.
"""

IMPLEMENT_SUFFIX: Final[str] = """

A design has already been settled. Implement it exactly as written:

{design}

Write the finished strategy as a single Python code block fenced with ```python. Put nothing after
the code block.
"""


class GraphOfThoughts:
    """Three independent proposals, two aggregations, one merge, one implementation."""

    name = "G6_graph_of_thoughts"

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

        # Identical prompt, three different seeds. The proposals differ because sampling is
        # stochastic, not because they were each asked a narrower question — asking three different
        # questions would rebuild the chain this arm exists to be an alternative to.
        proposal = PROPOSAL_PROMPT.format(theme=theme)
        a = recorder.ask(proposal, stage=1)
        b = recorder.ask(proposal, stage=2)
        c = recorder.ask(proposal, stage=3)

        agreement = recorder.ask(
            AGGREGATE_AGREEMENT_PROMPT.format(a=a, b=b, c=c), stage=4
        )
        strength = recorder.ask(
            AGGREGATE_STRENGTH_PROMPT.format(a=a, b=b, c=c), stage=5
        )
        design = recorder.ask(
            MERGE_PROMPT.format(agreement=agreement, strength=strength), stage=6
        )

        outcomes: list[str] = []
        prompt = build_prompt(theme) + IMPLEMENT_SUFFIX.format(design=design)
        source = attempt_code(recorder, prompt, stage=8, outcomes=outcomes)
        return recorder.finish(theme, source, outcomes)
