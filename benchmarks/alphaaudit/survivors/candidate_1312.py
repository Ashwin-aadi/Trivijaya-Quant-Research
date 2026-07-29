from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when a stock that has moved significantly in one direction "
        "is likely to move back towards its mean price. This strategy aims to identify such "
        "over-moved stocks and take positions against them."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the trailing average close price
        avg_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("avg_close"))
            .select(["symbol", "avg_close"])
        )

        # Join with latest closes to get recent prices and calculate reversion score
        closes = view.closes(lookback=self._window)
        merged = history.join(avg_close, on="symbol", how="inner")
        merged = (
            merged.with_columns(
                (pl.col("adj_close") - pl.col("avg_close")).abs().alias("reversion_score"),
                (pl.col("adj_close") / pl.col("close").shift(1) - 1.0).alias("r"),
            )
            .sort("reversion_score", descending=True)
            .select(["symbol", "session_date", "reversion_score"])
        )

        if merged.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify the top reversion candidates
        top_reversions = merged.head(self._window)["symbol"].to_list()
        weight = 1.0 / len(top_reversions)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_reversions},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest