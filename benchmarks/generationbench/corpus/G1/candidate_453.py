from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform the market over the long term. By tilting "
        "our portfolio towards these low-volatility stocks, we aim to capture this premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        volatilities: list[float] = []
        
        for symbol in symbols:
            adj_closes = history[symbol].drop_nulls().to_list()
            daily_returns = [(adj_closes[i+1]/adj_closes[i] - 1.0) for i in range(len(adj_closes)-1)]
            volatility = (sum([r**2 for r in daily_returns])**0.5)
            volatilities.append(volatility)

        sorted_symbols = [s for _, s in sorted(zip(volatilities, symbols))]
        
        top_3_symbols = sorted_symbols[:3]
        if not top_3_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_3_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_3_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest