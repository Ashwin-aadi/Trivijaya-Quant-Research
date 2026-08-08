from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting is a strategy that seeks to exploit the profitability of "
        "low-volatility stocks by identifying and investing in them. Over time, such stocks "
        "tend to outperform high-volatility counterparts."
    )

    def __init__(self, window: int = 30, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the daily returns
            returns = [(values[i] - values[i-1]) / values[i-1] if i > 0 else 0.0 for i in range(len(values))]
            mean_return = sum(returns) / len(returns)
            squared_deviations = [(r - mean_return)**2 for r in returns]
            volatility = (sum(squared_deviations) / self._window) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = sorted(volatilities, key=volatilities.get)[:self._top_n]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest