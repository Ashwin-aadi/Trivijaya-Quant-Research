"""The six arms P4 actually runs, and how to build each one.

One place that names the arms, so the runner, the progress reader and the verifier cannot disagree
about what the experiment consists of. A disagreement between them would show up as a missing arm
rather than as an error.

**Two arms named in the original design are retired and are not here:**
:mod:`src.generate.paradigms.multi_agent` and :mod:`src.generate.paradigms.evolutionary`, replaced
on 2026-08-04 by graph of thoughts and Monte Carlo tree search respectively, before any strategy was
generated. Their modules remain on disk and remain tested; they are simply not arms.

**Only G7 takes a fitness function**, and building it starts a worker process holding the price
panel. The factory therefore constructs it lazily, so a status query or a G1 run does not pay for a
backtester it will never use.
"""

from __future__ import annotations

from typing import Final

from src.generate.paradigms.base import Paradigm, ParadigmError
from src.generate.paradigms.cot import ChainOfThought
from src.generate.paradigms.graph_of_thoughts import GraphOfThoughts
from src.generate.paradigms.mcts import MonteCarloTreeSearch
from src.generate.paradigms.plain import PlainPrompting
from src.generate.paradigms.planning import Planning
from src.generate.paradigms.reflection import Reflection

#: Short name -> full arm name. The short name is what the PI types; the full name is what lands in
#: every record, so a corpus file can never be traced to the wrong arm.
ARMS: Final[dict[str, str]] = {
    "G1": "G1_plain",
    "G2": "G2_cot",
    "G4": "G4_planning",
    "G5": "G5_reflection",
    "G6": "G6_graph_of_thoughts",
    "G7": "G7_mcts",
}

#: Model calls issued per draw, retries excluded. Used only for the progress reader's estimate;
#: nothing scientific rests on it, and the measured token counts replace it in every report.
CALLS_PER_DRAW: Final[dict[str, int]] = {
    "G1": 1, "G2": 1, "G4": 4, "G5": 3, "G6": 7, "G7": 24,
}


def build(short_name: str) -> tuple[Paradigm, object | None]:
    """Construct one arm. Returns the paradigm and its fitness function, if it has one.

    The fitness is returned rather than hidden inside the paradigm so the runner can close its
    worker process and record how often the search had no signal to search on.
    """
    if short_name not in ARMS:
        raise ParadigmError(
            f"unknown arm {short_name!r}; the experiment consists of {', '.join(ARMS)}"
        )
    if short_name == "G1":
        return PlainPrompting(), None
    if short_name == "G2":
        return ChainOfThought(), None
    if short_name == "G4":
        return Planning(), None
    if short_name == "G5":
        return Reflection(), None
    if short_name == "G6":
        return GraphOfThoughts(), None

    # G7 alone scores its own intermediates. Imported here so that no other arm's run starts a
    # backtest worker or loads the price panel.
    from src.generate.paradigms.fitness import NetSharpeFitness

    fitness = NetSharpeFitness()
    return MonteCarloTreeSearch(fitness), fitness
