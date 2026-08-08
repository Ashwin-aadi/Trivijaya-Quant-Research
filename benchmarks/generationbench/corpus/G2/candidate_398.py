from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "High volatility can signal increased market uncertainty and potential for large price "
        "movements. Trend following strategies aim to capture these movements by scaling position "
        "sizes based on recent volatility. High volatilities suggest larger positions, while low "
        "volatility indicates caution."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            (closes["close"] / closes["close"].shift(1) - 1.0).abs()
            .rolling_sum(self._window)
            .mean()
            .to_list()
        )

        trends = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._window:
                continue

            last_close = float(closes["close"].tail(1)[symbol])
            mean_volatility = sum(volatility[-self._window:]) / self._window
            trend_weight = (last_close - recent_closes[0]) / (
                max(recent_closes) - min(recent_closes)
            ) * 2.0 + 0.5

            trends[symbol] = mean_volatility * trend_weight

        sorted_trends = sorted(trends.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, v in sorted_trends[:5]]
        weight_per_symbol = 1.0 / len(top_symbols)

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight_per_symbol for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest