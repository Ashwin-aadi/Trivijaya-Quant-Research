from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to capture trends by smoothing price changes and "
        "scaling them by historical volatility. High positive signals indicate a strong "
        "trending market, while low or negative signals suggest consolidation or reversal."
    )

    def __init__(self, window: int = 20, trend_window: int = 10) -> None:
        self._window = window
        self._trend_window = trend_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            price_changes = [float(v) for v in (closes[symbol].drop_nulls() / closes[symbol].shift(1).drop_nulls()) - 1.0].sort().to_list()
            mean_change = sum(price_changes[-self._trend_window:]) / self._trend_window
            std_deviation = (sum((x - mean_change) ** 2 for x in price_changes[-self._trend_window:]) / self._trend_window) ** 0.5

            if std_deviation > 0:
                scaled_trend = mean_change / std_deviation
                trends[symbol] = max(-1.0, min(1.0, scaled_trend))

        positive_symbols = [s for s, t in trends.items() if t >= 0]
        weight = 2.0 / len(positive_symbols) if positive_symbols else 0.0

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in positive_symbols} | {s: 0.0 for s in view.symbols if s not in positive_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest