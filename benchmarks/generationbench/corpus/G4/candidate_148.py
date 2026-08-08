from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy exploits the economic phenomenon where stock prices tend to revert to "
        "their historical price levels over time. Deviations from a trailing moving average are "
        "identified and traded back towards the mean, expecting a reversion."
    )

    def __init__(self, window: int = 20, threshold: float = 3.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma_column = f"sma_{self._window}"
        std_column = f"std_{self._window}"

        # Calculate 20-day Simple Moving Average (SMA)
        sma = (
            history.group_by("symbol")
                   .agg((pl.col("adj_close").mean()).alias(sma_column))
                   .with_columns(
                       (pl.col("adj_close") / pl.col(sma_column) - 1.0).alias("deviation")
                   )
        )

        # Calculate Standard Deviation of the 20-day closing prices
        std = (
            history.group_by("symbol")
                    .agg((pl.col("adj_close").std()).alias(std_column))
                    .select(pl.col("symbol"), sma_column, std_column)
        )

        merged = sma.join(std, on="symbol", how="inner")

        # Filter out symbols with insufficient history
        if merged.is_empty():
            return Signal(information_available_at=stamp, weights={})

        candidates = (
            merged.with_columns(
                (pl.col("deviation") > self._threshold) | (pl.col("deviation") < -self._threshold)
            )
                   .filter((pl.col("deviation") > self._threshold) | (pl.col("deviation") < -self._threshold))
        )

        if candidates.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Rank symbols based on absolute deviation from SMA
        ranked = (
            candidates.sort(pl.col("deviation").abs(), descending=True)
                       .with_columns((pl.arange(1, pl.count() + 1) / len(candidates)).alias("rank"))
        )

        top_n = min(len(ranked), 50)
        picks = [symbol for symbol in ranked["symbol"].to_list()[:top_n]]
        weight = 1.0 / len(picks)

        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest