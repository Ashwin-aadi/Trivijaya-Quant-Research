from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting our portfolio towards low-volatility stocks, we aim to capture this premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        for symbol in view.symbols:
            adj_close_series = history.filter(pl.col("symbol") == symbol)["adj_close"]
            returns = (adj_close_series / adj_close_series.shift(1) - 1.0).drop_nulls()
            volatility = returns.std().to_f64()
            volatilities.append(volatility)

        sorted_symbols = [symbol for _, symbol in sorted(zip(volatilities, view.symbols))]
        top_n_symbols = sorted_symbols[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())["session_date"].item()
    assert isinstance(newest, date)
    return newest