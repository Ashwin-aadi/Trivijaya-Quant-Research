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

        # Calculate the mean price and standard deviation for each stock
        mean_prices = (
            history.group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("mean"),
                (pl.col("adj_close").std()).alias("std"),
            )
            .collect()
        )

        # Filter out symbols that have no enough data points in the lookback period
        mean_prices = mean_prices.filter(pl.col("symbol").is_in(view.symbols))

        if mean_prices.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Join the mean prices back to history
        history_with_mean = (
            history.join(mean_prices, on="symbol", how="inner")
        )

        # Calculate the z-score for each closing price in the window
        history_with_mean = (
            history_with_mean.with_columns(
                (pl.col("adj_close") - pl.col("mean")) / pl.col("std").alias("zscore")
            )
        )

        # Identify symbols with a z-score outside the threshold
        reversion_candidates = (
            history_with_mean.filter(pl.col("zscore").abs() > self._threshold)
        )

        if reversion_candidates.height == 0:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = reversion_candidates["symbol"].to_list()

        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest