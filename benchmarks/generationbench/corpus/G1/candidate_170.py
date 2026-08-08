from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends while adjusting positions based on recent volatility. "
        "High volatilities may reduce exposure or reverse signals to limit risk, whereas low volatilities allow for more aggressive entry."
    )

    def __init__(self, window: int = 20, trend_weight: float = 1.0, vol_window: int = 10) -> None:
        self._window = window
        self._trend_weight = trend_weight
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._vol_window - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("symbol").alias("close")).pivot(index="session_date", columns="symbol")
        latest_closes = view.closes().select(pl.col("symbol").alias("latest_close"))

        # Calculate returns
        returns = (history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        ).sort("session_date").tail(self._window)["r"].to_list())

        # Calculate volatility
        volatilities = history.groupby("symbol").agg(
            (pl.col("r").std().alias("volatility"))
        ).select(pl.col("symbol", "volatility"))

        # Scale trend by volatility
        scaled_trends = [returns[i] * volatilities.get_column("volatility")[i] for i in range(len(returns))]

        # Determine signals based on the last return and volatility
        signal_symbols: list[str] = []
        for symbol in view.symbols:
            if (scaled_trend := scaled_trends[-1].get(symbol, 0.0)) > 0:
                signal_symbols.append(symbol)

        weights: dict[str, float] = {s: self._trend_weight / len(signal_symbols) for s in signal_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest