from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in the Indian market can be influenced by various factors such as "
        "holidays, fiscal years, and specific economic events. This strategy aims to identify "
        "overvalued or undervalued stocks based on historical seasonal trends."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

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

            # Calculate the mean close price and the seasonality factor
            mean_close = sum(values[-self._window:]) / self._window
            last_value = values[-1]
            seasonality_factor = last_value / mean_close - 1.0
            seasonality_factors[symbol] = seasonality_factor

        # Identify symbols with significant positive and negative seasonal trends
        top_gainers = sorted(seasonality_factors.items(), key=lambda x: abs(x[1]), reverse=True)[: self._top_n]
        top_sufferers = sorted(seasonality_factors.items(), key=lambda x: abs(x[1]))[: self._top_n]

        # Determine the final signal based on the balance of gainers and sufferers
        if len(top_gainers) > len(top_sufferers):
            picks = [symbol for symbol, _ in top_gainers]
        else:
            picks = [symbol for symbol, _ in top_sufferers]

        weights: dict[str, float] = {}
        if picks:
            weight = 1.0 / len(picks)
            weights = {p: weight for p in picks}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest