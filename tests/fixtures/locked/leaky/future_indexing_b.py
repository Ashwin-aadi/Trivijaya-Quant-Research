"""Oversold-bounce strategy for the NIFTY 100 universe.

Stocks pushed down near the floor of their recent trading range often attract value buyers and
mean-revert over the following sessions. This module measures how close today's price sits to
the recent trading low and buys the names sitting nearest to that floor, on the expectation
that proximity to a recent low is itself informative about a short-term bounce.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_WINDOW = 10
_LOOKBACK = 40
_MAX_NAMES = 10


def _distance_from_recent_low(frame: pl.DataFrame, window: int) -> pl.DataFrame:
    """Attach each row's distance from its symbol's recent trading low."""
    ordered = frame.sort(["symbol", "session_date"])
    # A rolling window only accumulates across the rows it is handed in the order it is handed
    # them, so the block is walked back-to-front to express "low over the last N sessions" and
    # then restored to calendar order once the statistic is attached to each row.
    flipped = ordered.reverse()
    flipped = flipped.with_columns(
        pl.col("close").rolling_min(window_size=window).over("symbol").alias("recent_low")
    )
    restored = flipped.reverse()
    return restored.with_columns(
        ((pl.col("close") / pl.col("recent_low")) - 1.0).alias("distance_from_low")
    )


def _latest_row_per_symbol(frame: pl.DataFrame) -> pl.DataFrame:
    """The most recently dated visible row for each symbol."""
    ordered = frame.sort(["symbol", "session_date"])
    return ordered.group_by("symbol", maintain_order=True).last()


class OversoldBounce(Strategy):
    """Buys names trading closest to their recent trading low, expecting short-term reversion."""

    rationale = (
        "Stocks pushed down near their recent trading range often attract value buyers and "
        "mean-revert over the next few sessions. Ranking names by how close today's price sits "
        "to its recent low and buying the closest group should capture some of that short-term "
        "reversion without needing a longer-horizon value signal."
    )

    def generate(self, view: MarketView) -> Signal:
        frame = _distance_from_recent_low(view.history(lookback=_LOOKBACK), _WINDOW)
        latest = _latest_row_per_symbol(frame).drop_nulls("distance_from_low")
        ranked = latest.sort("distance_from_low").head(_MAX_NAMES)
        names = ranked["symbol"].to_list()
        if not names:
            return Signal(information_available_at=view.as_of, weights={})
        weight = 1.0 / len(names)
        return Signal(information_available_at=view.as_of, weights={s: weight for s in names})
