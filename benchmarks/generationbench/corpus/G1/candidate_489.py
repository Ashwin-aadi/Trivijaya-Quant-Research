from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform the market over long horizons. By "
        "allocating more weight to these stocks, we aim to capture this effect."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities = {
            symbol: float(
                pl.col(symbol).std()
            ) for symbol in view.symbols if symbol in closes.columns
        }

        sorted_symbols = [
            s for _, s in sorted(volatilities.items(), key=lambda item: item[1])
        ]
        top_low_volatility = sorted_symbols[:5]

        weight = 0.2 / len(top_low_volatility)
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