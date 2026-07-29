from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy combines short-term and long-term momentum indicators to identify "
        "overbought and oversold conditions. Short-term momentum can signal a potential reversal,"
        " while long-term momentum provides a trend-following signal."
    )

    def __init__(self, short_window: int = 10, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._short_window, self._long_window))
        if closes.height < max(self._short_window, self._long_window):
            return Signal(information_available_at=stamp, weights={})

        short_moments: list[str] = []
        long_moments: list[str] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < max(self._short_window, self._long_window):
                continue

            short_momentum = (values[-1] - values[-self._short_window]) / values[
                -self._short_window
            ]
            long_momentum = (values[-1] - values[-self._long_window]) / values[
                -self._long_window
            ]

            if short_momentum > 0.5 and long_momentum > 0.3:
                short_moments.append(symbol)
            elif short_momentum < -0.5 and long_momentum < -0.3:
                long_moments.append(symbol)

        short_weight = 1.0 / len(short_moments) if short_moments else 0
        long_weight = 1.0 / len(long_moments) if long_moments else 0

        weights: dict[str, float] = {}
        for symbol in short_moments:
            weights[symbol] = short_weight * 0.6 + long_weight * 0.2
        for symbol in long_moments:
            weights[symbol] = short_weight * 0.4 + long_weight * 0.8

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest