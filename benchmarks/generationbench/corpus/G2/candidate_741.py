from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower downside risk and potentially higher "
        "returns. By tilting our portfolio towards low volatility stocks, we aim to reduce "
        "risk while still achieving competitive returns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol]) < self._window:
                continue
            close_series = [float(v) for v in history[symbol].to_list()]
            volatility = ((pl.Series(close_series).rolling_std(window=self._window)).item(0))
            volatilities[symbol] = volatility

        sorted_symbols = [
            s for _, s in sorted(volatilities.items(), key=lambda item: item[1])
        ][:5]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest