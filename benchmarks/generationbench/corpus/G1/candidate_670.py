from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit strong seasonal patterns. "
        "By identifying these patterns, we can construct a strategy that takes advantage of them."
    )

    def __init__(self, window: int = 60, seasonality_window: int = 365) -> None:
        self._window = window
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._seasonality_window)

        if closes.height < self._seasonality_window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._seasonality_window:
                continue

            # Calculate the seasonal factor by comparing recent close to historical close at same date last year
            recent_close = values[-1]
            same_date_last_year = next(
                (
                    float(v)
                    for v in closes[symbol][
                        (closes["session_date"] == stamp - pl.duration(years=1)) &
                        (closes["symbol"] == symbol)
                    ]["adj_close"].to_list()
                    if not pl.col("adj_close").is_null()
                ),
                recent_close,
            )
            seasonality_factors[symbol] = recent_close / same_date_last_year

        # Identify symbols with the strongest seasonal trend
        top_symbols = sorted(seasonality_factors, key=seasonality_factors.get, reverse=True)[:5]

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