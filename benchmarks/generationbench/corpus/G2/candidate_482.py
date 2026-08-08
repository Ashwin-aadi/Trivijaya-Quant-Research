from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the relationship between historical price "
        "volatility and future returns. Higher volatility often precedes a reversal in trend, "
        "and by scaling our position based on volatility, we can capture gains from both rising "
        "and falling markets."
    )

    def __init__(self, window: int = 50, scaling_factor: float = 2.0) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.sort("session_date")
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"),
            )
            .select(pl.col("symbol"), "session_date", "r")
            .drop_nulls()
        )

        # Calculate volatility
        vol = returns.groupby("symbol").agg(
            (pl.col("r").std().alias(f"volatility"))
        ).collect()

        # Determine trend direction based on most recent return
        latest_returns = history.sort("session_date").tail(self._window + 1)
        latest_return = (
            latest_returns.select(pl.col("symbol"), "r")
            .drop_nulls()
            .groupby("symbol")
            .agg([pl.col("r").last().alias("latest_return")])
            .collect()
        )

        # Merge volatility and return
        merged = vol.join(latest_return, on="symbol", how="inner")

        # Scale trend following signal based on volatility
        trend_signal = (
            (merged.select(pl.col("latest_return")) / pl.col(f"volatility").cast(float))
            * self._scaling_factor
        ).to_list()

        if not trend_signal:
            return Signal(information_available_at=stamp, weights={})

        # Create weights based on signal direction
        positive_symbols = [symbol for symbol, sig in zip(merged["symbol"], trend_signal) if sig > 0]
        weight = 1.0 / len(positive_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in positive_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest