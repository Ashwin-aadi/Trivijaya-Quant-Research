from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is due to risk premium compensation and behavioral biases that lead investors to "
        "overpay for volatility."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            price_changes = (history[symbol].drop_nulls().to_list()[1:] - 
                             [float(history[symbol][0])] * len(history))
            volatility = ((sum(v ** 2 for v in price_changes) / self._window) ** 0.5)
            volatilities[symbol] = volatility

        sorted_symbols = [k for k, v in sorted(volatilities.items(), key=lambda item: item[1])]
        top_low_volatility = sorted_symbols[:min(5, len(sorted_symbols))]
        
        if not top_low_volatility:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_low_volatility)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_low_volatility}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest