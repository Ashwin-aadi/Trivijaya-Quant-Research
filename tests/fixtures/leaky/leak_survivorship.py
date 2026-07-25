"""Cheat: restricts the universe to names that were still in the index at the end of the period.

The strategy trades an ordinary equal-weighted portfolio. Its defect is upstream of any trading
rule: it selects which stocks to consider using end-of-period membership, so every company that
was liquid in 2020 and had collapsed by 2024 is invisible to it. Only survivors are ever bought.

This is the bias the whole data layer was built to avoid, reintroduced deliberately. It is the
most damaging of the three because it needs no clever code and leaves no trace in the trading
logic — the cheat is entirely in the universe construction, which is why an auditor that only
inspects signal computation will miss it.

Expected behaviour: a persistently flattering result, driven by the absence of losers rather than
the presence of winners.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class LeakSurvivorship(Strategy):
    """Equal-weights only those names present in the final universe snapshot."""

    rationale = (
        "Broad exposure to the liquid large-cap segment, equally weighted to avoid concentration "
        "in any single name."
    )

    def __init__(self, universe: pl.DataFrame) -> None:
        # THE CHEAT: the constituent list is read from the LAST rebalance in the whole history and
        # then applied backwards to every date. Names that dropped out along the way never appear.
        final_rebalance = universe["rebalance_date"].max()
        self._survivors: frozenset[str] = frozenset(
            universe.filter(pl.col("rebalance_date") == final_rebalance)["symbol"].to_list()
        )

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        # Intersecting the point-in-time universe with the final one discards exactly the names
        # that failed — which is what makes the backtest look better than reality was.
        eligible = [s for s in view.symbols if s in self._survivors]
        if not eligible:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(eligible)
        return Signal(information_available_at=stamp, weights=dict.fromkeys(eligible, weight))


def _latest_visible(view: MarketView) -> date:
    history = view.history()
    if history.is_empty():
        return date(1900, 1, 1)
    latest = history["session_date"].max()
    assert isinstance(latest, date)
    return latest
