from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows the logic of trend-following but scales trades by volatility. "
        "During periods of high volatility, trades are scaled down to avoid excessive risk, and during low volatility, trades are increased to capitalize on stable trends."
    )

    def __init__(self, window: int = 20, vol_window: int = 15) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        vol_history = view.history(lookback=self._vol_window)
        close_series = [float(v) for v in closes.to_list()]
        vol = (pl.Series(close_series).rolling_std(window=self._vol_window)).item()

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        weights: dict[str, float] = {}
        for symbol in symbols:
            close_series_symbol = closes[symbol].to_list()
            trend_score = sum(
                (close - min(close_series_symbol)) / (max(close_series_symbol) - min(close_series_symbol))
                for close in close_series_symbol
            ) / len(close_series_symbol)

            if vol > 1.0:  # High volatility, scale down trade size
                weights[symbol] = max(0.01, trend_score * 0.2)
            else:  # Low volatility, scale up trade size
                weights[symbol] = max(0.01, trend_score * 2.0)

        if not any(weights.values()):
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(weights.values())
        for symbol in symbols:
            if symbol in weights:
                weights[symbol] /= total_weight
            else:
                weights[symbol] = 0

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest