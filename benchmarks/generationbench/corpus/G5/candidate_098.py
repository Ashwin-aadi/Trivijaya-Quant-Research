from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by scaling trades based "
        "on historical volatility. High volatility periods are expected to be more profitable."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(sym) for sym in view.symbols]
        mean_returns = (
            (history["close"] / history["close"].shift(1) - 1.0).mean().to_list()
        )

        if len(mean_returns) < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_return = sum(mean_returns) / len(mean_returns)
        volatility = (
            history.select([pl.col("symbol"), pl.col("close")])
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").std().over(pl.col("session_date")) / self._vol_window).alias("volatility")
            )
            .select(pl.col("volatility"))
        )

        if volatility.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility = float(volatility["volatility"].mean())
        weight = (mean_return / volatility) if volatility > 0 else 1.0

        selected_symbols = symbols
        return Signal(
            information_available_at=stamp,
            weights={sym: weight for sym in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest