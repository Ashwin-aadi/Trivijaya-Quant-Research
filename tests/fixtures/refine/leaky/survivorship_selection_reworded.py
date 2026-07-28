"""Category ``survivorship_selection``, variant ``reworded``.

Defect: identical to ``survivorship_selection_obvious``. The membership frame is reduced to the
single newest rebalance it contains, and that one set of names is then applied to every earlier
decision date, so companies removed from the index part-way through the period are never held and
never lose money.

Nothing in the code body names the defect. The constructor argument is not called a universe, the
stored set is not called survivors, and the newest rebalance date is not called final. The
structure is unchanged: a maximum over a date column, an equality filter against that maximum, and
a membership test applied to dates that precede it. A detector that needs the word to find the
defect has learned nothing about the defect.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class RosterEqualWeight(Strategy):
    """Equal-weights the names carried on the reference roster."""

    rationale = (
        "Broad exposure to the large-cap segment. Holding the index constituents equally weighted "
        "avoids taking a view on any single name and keeps turnover low, which matters once "
        "Indian statutory costs are applied to every leg."
    )

    def __init__(self, reference_set: pl.DataFrame) -> None:
        marker = reference_set["rebalance_date"].max()
        self._roster: frozenset[str] = frozenset(
            reference_set.filter(pl.col("rebalance_date") == marker)["symbol"].to_list()
        )

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        bucket = [s for s in view.symbols if s in self._roster]
        return Signal(information_available_at=stamp, weights=_allocate(bucket))


def _stamp(view: MarketView) -> date:
    """Most recent session in the visible window."""
    seen = view.history()
    if seen.is_empty():
        return date(1900, 1, 1)
    newest = seen["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _allocate(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
