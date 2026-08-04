"""Generation paradigms: the experimental subjects of P4, measured through the frozen stack.

Each module here implements one procedure for turning P1's task specification into an executable
strategy. The *model* is held fixed across all of them; only the procedure varies. That is the
whole design — which paradigms work is a question that outlives the next four model generations,
whereas which vendor led in a given month does not.

**G3, retrieval-augmented generation, is not implemented.** It requires a corpus of factor
literature to retrieve from, and this repository holds none: `data/raw/` contains bhavcopy prices,
calendars, participant flows and the SEBI event timeline, and no embedding dependency is installed.
Acquiring and indexing a literature corpus is a project, not a paradigm arm, and the market-data
budget is Rs 0. It is dropped and recorded as dropped, before any data existed, rather than being
attempted badly.

**Two arms were retired on 2026-08-04, before any strategy was generated**, by PI ruling:
:mod:`multi_agent` was replaced by :mod:`graph_of_thoughts`, and :mod:`evolutionary` by
:mod:`mcts`. The retired modules remain on disk and remain tested, so the swap is inspectable
rather than an absence, but they are not arms. :mod:`registry` is the single place that says what
the experiment consists of.

The paradigm names keep their charter numbering so the gaps are visible rather than tidied away.
"""

from src.generate.paradigms.base import Draw, Paradigm, ParadigmError
from src.generate.paradigms.registry import ARMS

__all__ = ["ARMS", "Draw", "Paradigm", "ParadigmError"]
