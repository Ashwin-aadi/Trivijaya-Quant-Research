from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Seasonal effects can be a significant driver of equity returns. By identifying "
        "historical patterns, we can exploit these effects to generate trading signals."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s.symbol) for s in view.symbols]
        mean_closes = closes[symbols].mean(axis=0)

        seasonality_factors = {}
        for symbol in symbols:
            if symbol not in mean_closes.columns:
                continue
            values = [float(v) for v in mean_closes[symbol].to_list()]
            max_close = max(values)
            factor = (values[-1] / max_close - 1.0) * 100.0
            seasonality_factors[symbol] = factor

        sorted_symbols = sorted(seasonality_factors, key=lambda x: seasonality_factors[x], reverse=True)[:5]
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