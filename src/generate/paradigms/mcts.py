"""G7 — Monte Carlo tree search over strategy design decisions.

Replaces the evolutionary arm by PI ruling of 2026-08-04, before any strategy was generated. The
retired module is kept at :mod:`src.generate.paradigms.evolutionary` and is not run.

**What is searched.** Not strategies, but *designs*. The tree's root is the theme; a node at depth
one has chosen a signal, at depth two a portfolio rule, at depth three a risk rule. A node is
therefore a partial specification, and a path from root to depth three is a complete one.

**Why the cost is what it is.** A partial design cannot be scored — only a runnable strategy can.
Every iteration must therefore *complete* the design it selected into code before it learns
anything, so an iteration costs one short expansion call plus one full implementation call. This is
the reason MCTS is not cheaper than population breeding at equal search depth, and it was estimated
and reported before the arm was approved rather than discovered during the run.

**The loop-closing property is what RQ4 needs.** Like the arm it replaces, this one produces a
single strategy by making and discarding many, and under an honest ledger it is charged for every
one. Whether the search survives that charge is the question. An arm that counted one trial per
returned strategy would answer it dishonestly and by construction.

**The fitness is injected and is backtest-only**, net of costs by PI ruling — see
:mod:`src.generate.paradigms.fitness`. Nothing in the frozen evaluation stack is reachable from the
search, and neither is the holdout.

Reference: Kocsis & Szepesvári, *Bandit based Monte-Carlo Planning*, ECML 2006 — UCT, equation for
the selection score in §2. Rewards there are bounded in [0,1]; Sharpe is not, so scores are
min-max normalised against everything the tree has seen (§ ``_uct``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from src.audit.semantic import MODEL_TAG, OLLAMA_HOST
from src.generate.paradigms.base import CallRecorder, Draw, ParadigmError, attempt_code
from src.generate.paradigms.fitness import Fitness
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import TokenAccount

#: UCT exploration constant. sqrt(2) is the standard value for rewards normalised to [0,1] and is
#: not tuned here: tuning it against the corpus this arm is measured on would be selecting a
#: hyperparameter on the outcome, which is the pathology this lab exists to detect.
UCT_C: Final[float] = math.sqrt(2.0)

#: What is decided at each depth, in order. Three layers, because a strategy specification has
#: roughly three independent choices in it and a deeper tree would spend the whole budget before
#: any path reached a runnable design.
DECISION_LAYERS: Final[tuple[str, ...]] = (
    "the signal: what is measured, from which of daily open, high, low, close and volume, and "
    "over what lookback",
    "the portfolio rule: when a position is opened and closed, how many names are held, and how "
    "they are weighted",
    "the risk rule: what bounds the loss on a position and on the portfolio, and what would stop "
    "the strategy trading",
)

DECISION_PROMPT: Final[str] = """A systematic equity strategy is being designed for the Indian
market, on this theme: {theme}.

{settled}

Decide {layer}.

In at most 120 words, and without writing code, state one specific choice. Commit to it. Do not
list alternatives and do not decide anything else.{avoid}
"""

SETTLED_NONE: Final[str] = "Nothing has been decided yet."

SETTLED_SOME: Final[str] = """These decisions are already settled and must not be changed:

{decisions}"""

AVOID_SUFFIX: Final[str] = """

These choices have already been considered at this step. Propose a genuinely different one:

{siblings}"""

COMPLETE_SUFFIX: Final[str] = """

The following design decisions have been made and must be implemented as stated:

{decisions}

Where the design is incomplete, choose sensibly and keep it simple. Write the finished strategy as
a single Python code block fenced with ```python. Put nothing after the code block.
"""


@dataclass
class _Node:
    """One partial design: the decisions made on the path from the root to here."""

    parent: int | None
    depth: int
    #: The decision text chosen at this node. Empty at the root.
    decision: str = ""
    children: list[int] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0


class MonteCarloTreeSearch:
    """Search the space of strategy designs, returning the best strategy the search found.

    ``iterations`` is the whole budget: each one issues exactly one short expansion call and one
    implementation call, so the arm's cost is linear in it and capping it caps the arm. That is the
    property the PI selected this paradigm for, and it is the property population breeding does not
    have, where cost is ``population x generations`` and the two interact.
    """

    name = "G7_mcts"

    def __init__(
        self,
        fitness: Fitness,
        *,
        iterations: int = 12,
        branching: int = 2,
        exploration: float = UCT_C,
    ) -> None:
        if iterations < 1:
            raise ParadigmError("iterations must be at least 1")
        if branching < 1:
            raise ParadigmError("branching must be at least 1")
        self.fitness = fitness
        self.iterations = iterations
        self.branching = branching
        self.exploration = exploration
        self.max_depth = len(DECISION_LAYERS)

    # ---- the search -------------------------------------------------------------------------

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
        nodes: list[_Node] = [_Node(parent=None, depth=0)]

        best_source, best_score = "", float("-inf")
        #: Any conforming source, kept so that a search whose fitness never returned a number is
        #: still judged usable by the same rule as every other arm. Without it this arm alone would
        #: be held to the fitness function's stricter standard and its yield would not be
        #: comparable.
        fallback = ""
        seen: list[float] = []
        expansions = 0

        for iteration in range(self.iterations):
            node = self._select(nodes, seen)
            if nodes[node].depth < self.max_depth:
                node = self._expand(recorder, nodes, node, theme, expansions)
                expansions += 1

            decisions = self._path(nodes, node)
            prompt = build_prompt(theme) + COMPLETE_SUFFIX.format(
                decisions=_numbered(decisions)
            )
            source = attempt_code(recorder, prompt, stage=200 + iteration, outcomes=outcomes)
            if source and not fallback:
                fallback = source

            score = self.fitness(source) if source else None
            if score is not None:
                seen.append(score)
                if score > best_score:
                    best_source, best_score = source, score
            # An unscoreable rollout still visited the path, and it must be credited as the worst
            # reward seen rather than as zero. Zero is not the bottom of this scale — Sharpe is
            # routinely negative — so crediting zero would make a branch that cannot produce
            # runnable code look *better* than one that produces losing strategies. Dropping the
            # visit instead would leave that branch looking unexplored and the search would return
            # to it forever.
            self._backpropagate(nodes, node, score if score is not None else _floor(seen))

        chosen = best_source or fallback
        return recorder.finish(theme, chosen, outcomes)

    # ---- the four MCTS operations ------------------------------------------------------------

    def _select(self, nodes: list[_Node], seen: list[float]) -> int:
        """Descend from the root by UCT while the current node is fully expanded."""
        current = 0
        while (
            nodes[current].depth < self.max_depth
            and len(nodes[current].children) >= self.branching
        ):
            current = max(
                nodes[current].children,
                key=lambda child: self._uct(nodes, child, nodes[current].visits, seen),
            )
        return current

    def _uct(self, nodes: list[_Node], child: int, parent_visits: int, seen: list[float]) -> float:
        """Upper confidence bound for a child, with rewards min-max normalised into [0, 1].

        An unvisited child is infinitely attractive, which is UCT's standard behaviour and is what
        guarantees every sibling is tried once before any is tried twice.
        """
        node = nodes[child]
        if node.visits == 0:
            return float("inf")
        mean = node.total_value / node.visits
        low, high = (min(seen), max(seen)) if seen else (0.0, 0.0)
        span = high - low
        # With one distinct score seen, every branch is equally good on the evidence; a constant
        # exploitation term leaves the exploration term to decide, which is the correct behaviour.
        exploit = 0.5 if span <= 0 else (mean - low) / span
        explore = self.exploration * math.sqrt(math.log(max(parent_visits, 1)) / node.visits)
        return exploit + explore

    def _expand(
        self,
        recorder: CallRecorder,
        nodes: list[_Node],
        parent: int,
        theme: str,
        expansion_number: int,
    ) -> int:
        """Add one child to ``parent`` by asking the model for the next design decision."""
        settled = self._path(nodes, parent)
        siblings = [nodes[c].decision for c in nodes[parent].children]
        prompt = DECISION_PROMPT.format(
            theme=theme,
            settled=SETTLED_NONE if not settled else SETTLED_SOME.format(
                decisions=_numbered(settled)
            ),
            layer=DECISION_LAYERS[nodes[parent].depth],
            avoid="" if not siblings else AVOID_SUFFIX.format(siblings=_numbered(siblings)),
        )
        decision = recorder.ask(prompt, stage=100 + expansion_number)
        nodes.append(_Node(parent=parent, depth=nodes[parent].depth + 1, decision=decision))
        child = len(nodes) - 1
        nodes[parent].children.append(child)
        return child

    def _backpropagate(self, nodes: list[_Node], node: int, reward: float) -> None:
        """Credit ``reward`` to every node on the path from ``node`` to the root."""
        current: int | None = node
        while current is not None:
            nodes[current].visits += 1
            nodes[current].total_value += reward
            current = nodes[current].parent

    def _path(self, nodes: list[_Node], node: int) -> list[str]:
        """The decisions from the root down to ``node``, in the order they were made."""
        decisions: list[str] = []
        current: int | None = node
        while current is not None and nodes[current].parent is not None:
            decisions.append(nodes[current].decision)
            current = nodes[current].parent
        return list(reversed(decisions))


def _floor(seen: list[float]) -> float:
    """The reward credited to a rollout that could not be scored: the worst score seen so far.

    Zero before anything has been scored, which is the only value available and is harmless because
    at that point every branch receives it.
    """
    return min(seen) if seen else 0.0


def _numbered(items: list[str]) -> str:
    return "\n\n".join(f"{i}. {text}" for i, text in enumerate(items, start=1))
