from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceReversion(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency for prices to revert to recent "
        "mean levels. This strategy identifies stocks that have moved far from their mean "
        "price over a lookback period and bets on them moving back."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean price and standard deviation
        means = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").mean().alias("mean"),
                 pl.col("adj_close").std().alias("std"))
            )
            .to_horizontal("symbol", "mean", "std")
        )

        # Calculate the z-score for each closing price
        means_with_zscore = (
            history.join(means, on="symbol", how="left")
            .with_column(
                (pl.col("adj_close") - pl.col("mean")) / pl.col("std").fill_null(1).alias("z_score")
            )
        )

        # Filter out symbols with insufficient data or too close to the mean
        filtered_history = means_with_zscore.select(["symbol", "session_date", "adj_close", "z_score"])
        filtered_history = filtered_history.filter(
            (pl.col("z_score") > self._threshold) | (pl.col("z_score") < -self._threshold)
        )

        # Find the latest session's symbol with high or low z-score
        latest_session = view.history(lookback=None).select(["symbol", "adj_close"])
        combined = filtered_history.join(latest_session, on="symbol", how="inner")

        if combined.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Determine the symbol to invest in based on the highest z-score
        target_symbol = (
            combined.sort("z_score", descending=True)
            .select(["symbol"])
            .head(1)["symbol"]
            .to_list()[0]
        )

        weight = 1.0
        return Signal(information_available_at=stamp, weights={target_symbol: weight})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest