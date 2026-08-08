from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit strong seasonality effects based on calendar "
        "events. We exploit these patterns by investing in those stocks during their historically "
        "strong periods."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_factors = {symbol: 0.0 for symbol in view.symbols}
        for date_str in closes["session_date"].to_list():
            month = int(date_str.split("-")[1])
            for symbol in view.symbols:
                if symbol not in seasonal_factors:
                    continue
                seasonal_factors[symbol] += 1 if month == 3 else 0

        sorted_symbols = sorted(seasonal_factors, key=lambda s: -seasonal_factors[s])
        top_n_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest