from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility ones over the long term. "
        "This strategy aims to tilt the portfolio towards low-volatility stocks to potentially enhance returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        for symbol in symbols:
            close_prices = [float(v) for v in history[symbol].to_list()]
            returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1] if i > 0 else 0.0 for i in range(len(close_prices))]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities.append(volatility)

        sorted_symbols = [s for _, s in sorted(zip(volatilities, symbols), key=lambda x: x[0])]
        weight = 1.0 / self._window
        return Signal(
            information_available_at=stamp,
            weights={sorted_symbols[i]: weight for i in range(self._window)},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest