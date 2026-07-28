"""Category ``survivorship_selection``, variant ``obvious``.

Defect: the tradable list is taken from the last rebalance in the whole membership history and then
applied to every earlier date. Companies that were index constituents in 2016 and had been removed
by 2023 are invisible to the strategy on every single day, including the days on which they were
perfectly tradable and were about to fall apart.

Written the direct way: the end-of-period membership is read in the constructor, stored under a
name that says what it is, and intersected with the point-in-time universe on each decision.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class SurvivorshipSelectionObvious(Strategy):
    """Equal-weights the names that were still in the index at the end of the period."""

    rationale = (
        "Broad exposure to the large-cap segment. Holding the index constituents equally weighted "
        "avoids taking a view on any single name and keeps turnover low, which matters once "
        "Indian statutory costs are applied to every leg."
    )

    def __init__(self, universe: pl.DataFrame) -> None:
        # THE CHEAT: the membership list is read from the LAST rebalance in the entire history and
        # then used on every date before it. Nothing that was later removed from the index can ever
        # be selected, so the backtest trades a basket assembled with hindsight.
        final_rebalance = universe["rebalance_date"].max()
        self._survivors: frozenset[str] = frozenset(
            universe.filter(pl.col("rebalance_date") == final_rebalance)["symbol"].to_list()
        )

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        eligible = [s for s in view.symbols if s in self._survivors]
        return Signal(information_available_at=stamp, weights=_spread(eligible))


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _spread(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
