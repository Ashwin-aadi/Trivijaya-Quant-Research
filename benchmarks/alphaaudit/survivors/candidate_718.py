from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTradeStrategy(Strategy):
    rationale = (
        "Certain stocks tend to exhibit stronger performance during specific times of the year. "
        "By identifying these patterns, we can capitalize on seasonal effects in the market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            month = stamp.month
            mean_price = sum(values) / len(values)
            seasonality_factors[symbol] = (values[month - 1] / mean_price - 1.0)

        sorted_factors = sorted(seasonality_factors.items(), key=lambda x: x[1], reverse=True)
        top_symbols, _ = zip(*sorted_factors)
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest