from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting our portfolio towards low-volatility stocks, we aim to benefit from this effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]

        for symbol in symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_close = sum(values) / len(values)
            variance = (
                0
                if all(value == values[0] for value in values)
                else sum((value - mean_close) ** 2 for value in values) / (len(values) - 1)
            )
            volatilities.append(variance)

        sorted_indices = [i[0] for i in sorted(enumerate(volatilities), key=lambda x: x[1])]
        top_n_symbols = [symbols[index] for index in sorted_indices[:5]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest