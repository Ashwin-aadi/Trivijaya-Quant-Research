from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the idea that during periods of high volatility, "
        "trends are more likely to persist. By scaling trends by their historical volatility, we can "
        "potentially capture profitable movements in the market."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate simple returns
        returns = closes.melt().pipe(
            lambda df: df.with_columns(
                (pl.col("value").shift(-1) / pl.col("value") - 1).alias("return")
            )
        ).select(["symbol", "session_date", "return"])

        # Calculate volatility for each symbol
        volatilities = returns.groupby("symbol").agg(
            pl.col("return").std().alias("volatility")
        )

        # Calculate trend signal (mean of returns)
        trends = (
            returns.groupby("symbol")
            .agg((pl.col("return") / pl.col("volatility").shift(-1)).alias("trend"))
            .select(["symbol", "session_date", "trend"])
        )

        combined = volatilities.join(trends, on="symbol")
        trends_and_volatilities = (
            combined.with_columns(
                (pl.col("trend") * pl.col("volatility").shift(-1)).alias("signal")
            )
            .sort("session_date", descending=True)
            .tail(20)
            .select(["symbol", "signal"])
            .to_dict(False)
        )

        # Select symbols with the highest signal
        picks = sorted(trends_and_volatilities, key=lambda x: x["signal"], reverse=True)[:5]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s["symbol"]: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest