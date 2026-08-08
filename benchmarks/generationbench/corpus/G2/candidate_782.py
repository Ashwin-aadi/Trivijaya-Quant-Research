from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Large price movements (high volatility) in a given direction may indicate "
        "momentum or market strength. By scaling trend following signals by recent "
        "volatility, we aim to capture more of the upside when markets are strong and "
        "reduce exposure during periods of high volatility."
    )

    def __init__(self, lookback: int = 60, threshold: float = 1.5) -> None:
        self._lookback = lookback
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (closes["adj_close"] / 2 + closes["open"] / 2).mean()
        vol = (
            ((closes["adj_close"] - mean_close) ** 2)
            .sum()
            .sqrt()  # Standard deviation
            / self._lookback**0.5
        )

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_values) < self._lookback:
                continue

            trend = (close_values[-1] - close_values[0]) / sum(
                abs(close_values[i] - close_values[i + 1])
                for i in range(len(close_values) - 1)
            )

            if trend > 0 and vol[symbol] * self._threshold < trend:
                weights[symbol] = 1.0

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest