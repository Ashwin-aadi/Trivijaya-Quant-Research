"""G7 — evolutionary: generate a population, score it, breed from the best, repeat. **RETIRED.**

**This arm is not run in P4.** It was built, tested and costed, and then replaced by
:mod:`src.generate.paradigms.mcts` under the PI ruling of 2026-08-04 — before a single strategy had
been generated, which is the only point at which dropping an arm is honest. The module is kept
rather than deleted so the retired design is inspectable and so the swap is a visible decision
rather than an absence.

Monte Carlo tree search inherits the property this arm existed to supply: it closes a loop with a
fitness signal, so RQ4 — whether an honest trial ledger erases the gains of iterating — survives the
swap unchanged. What it adds is a budget that is linear in one parameter, where this arm's cost is
``population x generations`` and the two interact.

**The fitness function is injected, and it must be backtest-only.** Nothing in the frozen evaluation
stack may be consulted during breeding. An arm that selected on the auditor would be optimising
against the instrument that later measures it. The concrete implementation lives in
:mod:`src.generate.paradigms.fitness`.

The fitness window is the development period. The holdout is not reachable from here, and must
never be made reachable.
"""

from __future__ import annotations

from typing import Final

from src.audit.semantic import MODEL_TAG, OLLAMA_HOST
from src.generate.paradigms.base import CallRecorder, Draw, ParadigmError, attempt_code
from src.generate.paradigms.fitness import Fitness
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import TokenAccount

__all__ = ["Evolutionary", "Fitness", "MUTATE_SUFFIX"]

MUTATE_SUFFIX: Final[str] = """

Here is a strategy that scored {score} on historical data:

```python
{parent}
```

Produce a variant of it. Change one thing that you believe limits it — a lookback, a ranking rule,
a sizing rule, or the signal itself — and keep everything else. State the change in the rationale.

Write the variant as a single Python code block fenced with ```python. Put nothing after the code
block.
"""


class Evolutionary:
    """Breed a population over generations, returning the fittest individual.

    Defaults give 4 + 2 x 4 = 12 individuals per returned strategy, so this arm is roughly an
    order of magnitude more expensive than plain prompting and carries an order of magnitude more
    trials. Both facts are measured rather than assumed: the token accountant supplies the first
    and ``candidates_evaluated`` the second.
    """

    name = "G7_evolutionary"

    def __init__(
        self, fitness: Fitness, *, population: int = 4, generations: int = 2, survivors: int = 2
    ) -> None:
        if survivors < 1 or survivors > population:
            raise ParadigmError("survivors must be between 1 and the population size")
        self.fitness = fitness
        self.population = population
        self.generations = generations
        self.survivors = survivors

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
        prompt = build_prompt(theme)

        scored = self._score(
            [
                attempt_code(recorder, prompt, stage=10 + i, outcomes=outcomes)
                for i in range(self.population)
            ]
        )
        for generation in range(1, self.generations + 1):
            scored = self._score(
                self._offspring(recorder, prompt, scored, outcomes, generation)
            ) + scored

        best = max(scored, key=lambda pair: pair[1], default=("", float("-inf")))
        # An empty source reaches finish() as "no code produced", which is the honest outcome when
        # nothing in the population was ever evaluable.
        return recorder.finish(theme, best[0] if best[1] > float("-inf") else "", outcomes)

    def _score(self, sources: list[str]) -> list[tuple[str, float]]:
        """Attach a fitness to each source, dropping those that cannot be scored at all."""
        scored: list[tuple[str, float]] = []
        for source in sources:
            if not source:
                continue
            value = self.fitness(source)
            if value is not None:
                scored.append((source, value))
        return scored

    def _offspring(
        self,
        recorder: CallRecorder,
        prompt: str,
        scored: list[tuple[str, float]],
        outcomes: list[str],
        generation: int,
    ) -> list[str]:
        """Mutate the fittest survivors back up to the population size.

        With nothing scorable to breed from, the generation is redrawn from the base prompt rather
        than skipped. Skipping would silently reduce this arm's trial count on exactly the draws
        where its search was going worst.
        """
        parents = sorted(scored, key=lambda pair: pair[1], reverse=True)[: self.survivors]
        children: list[str] = []
        for i in range(self.population):
            stage = 20 + generation * 10 + i
            if not parents:
                children.append(attempt_code(recorder, prompt, stage=stage, outcomes=outcomes))
                continue
            parent, score = parents[i % len(parents)]
            mutated = prompt + MUTATE_SUFFIX.format(parent=parent, score=f"{score:.4f}")
            children.append(attempt_code(recorder, mutated, stage=stage, outcomes=outcomes))
        return children
