from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often less affected by market downturns and can provide more stable returns. "
        "By tilting our portfolio towards low-volatility stocks, we aim to reduce overall portfolio risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not (view.symbols and len(closes.columns) == len(view.symbols)):
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            volatility = ((pl.Series(values).std()) ** 2)
            volatilities.append(volatility)

        sorted_symbols = [s for _, s in sorted(zip(volatilities, view.symbols))]
        top_symbols = sorted_symbols[:3]
        weight = 1.0 / len(top_symbols) if top_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest