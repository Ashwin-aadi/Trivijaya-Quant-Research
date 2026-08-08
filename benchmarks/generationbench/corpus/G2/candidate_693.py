from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "By tilting our portfolio towards low-volatility stocks, we can capture this premium."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Calculate volatility
        history = (
            history.with_columns(
                (pl.col("return") * pl.col("return")).sum().sqrt().alias("volatility")
            )
            .sort("volatility", descending=False)
            .select(["symbol", "volatility"])
            .head(10)  # Select top 10 low-volatility symbols
        )

        if history.height < 5:
            return Signal(information_available_at=stamp, weights={})

        volatilities = [float(v) for v in history["volatility"].to_list()]
        weights = [(1 - (v / max(volatilities))) * 0.2 for v in volatilities]
        total_weight = sum(weights)

        # Normalize weights if the sum is not close to one
        if abs(total_weight - 1) > 0.05:
            weights = [w / total_weight for w in weights]

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in zip(history["symbol"], weights)}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest