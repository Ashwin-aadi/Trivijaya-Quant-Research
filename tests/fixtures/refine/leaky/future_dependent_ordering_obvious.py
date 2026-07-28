"""Category ``future_dependent_ordering``, variant ``obvious``.

Defect: the ranking is computed once, over the whole period, and then applied to every decision
date. Each name's key is its total return from the first session of the sample to the last, so the
ordering is fixed by the outcome. The strategy holds the eventual winners from the first day it
trades, years before anything distinguished them.

The arithmetic contains no forward index and no fitted transform. Every individual number is a
legitimate return over a legitimate window. The defect is in what the ordering depends on: a
selection whose rank order is a function of the entire series is a selection made with hindsight,
however honestly each element of it was computed.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class FutureDependentOrderingObvious(Strategy):
    """Holds the names with the highest whole-period return."""

    rationale = (
        "Persistent compounders are rare and identifying them is the whole game in Indian large "
        "caps, where the index return is dominated by a handful of names over any long window. "
        "A concentrated book in the strongest compounders, rebalanced back to equal weight, is a "
        "defensible expression of that view."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        self._top_n = top_n
        # THE CHEAT: the ranking key is each name's return across the entire sample, first session
        # to last. Sorting on it produces an order that could only be known at the end, and that
        # order is then used on every date from the beginning.
        summary = (
            panel.sort(["symbol", "session_date"])
            .group_by("symbol")
            .agg(
                pl.col("adj_close").first().alias("start"),
                pl.col("adj_close").last().alias("end"),
            )
            .with_columns((pl.col("end") / pl.col("start") - 1.0).alias("key"))
            .sort("key", descending=True)
        )
        self._ordering: list[str] = [str(s) for s in summary["symbol"].to_list()]

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        tradable = set(view.symbols)
        picks = [symbol for symbol in self._ordering if symbol in tradable][: self._top_n]
        return Signal(information_available_at=stamp, weights=_spread(picks))


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
