from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "By tilting towards low-volatility stocks, we aim to capture this alpha."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        low_vol_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue

            mean_return = sum(adj_closes[i] / adj_closes[i - 1] - 1.0 for i in range(1, self._window)) / (self._window - 1)
            vol = ((adj_closes[-1] / adj_closes[0]) ** (252 / self._window) - 1) * 100
            if vol < 20:  # Assuming a threshold of 20% volatility
                low_vol_symbols.append(symbol)

        low_vol_symbols = low_vol_symbols[:5]  # Selecting top 5 symbols with lowest volatility
        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(low_vol_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in low_vol_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest