from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength (higher ratio of average price over the lookback period "
        "to recent price) are likely to continue their recent outperformance. This strategy aims to "
        "overweight these stocks in the portfolio."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average price over the window
        avg_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg"))
            .with_columns((pl.col("avg") / pl.col("adj_close")).alias("rs_ratio"))
        )

        # Filter symbols that meet the relative strength threshold
        filtered_rs = avg_close.filter(
            (pl.col("rs_ratio") > self._threshold)
            & (pl.col("rs_ratio").is_not_null())
        )
        if filtered_rs.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected symbol
        symbols = [str(row.symbol) for row in filtered_rs.rows()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest