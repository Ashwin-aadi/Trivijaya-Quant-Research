from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTradeStrategy(Strategy):
    rationale = (
        "Seasonality effects often result from predictable patterns in the market driven by "
        "events such as earnings reports or holidays. By identifying and trading on these patterns, "
        "we can potentially capitalize on consistent market behavior."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_signals = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the average close price over the lookback period
            avg_close = sum(values) / len(values)

            # Identify recent close above or below seasonal threshold
            threshold_high = avg_close * 1.05
            threshold_low = avg_close * 0.95
            if values[-1] >= threshold_high:
                seasonal_signals[symbol] = "buy"
            elif values[-1] <= threshold_low:
                seasonal_signals[symbol] = "sell"

        buy_signals = {s: 1 for s in seasonal_signals if seasonal_signals[s] == "buy"}
        sell_signals = {s: -1 for s in seasonal_signals if seasonal_signals[s] == "sell"}

        weight = 0.5
        buy_weights = {symbol: weight / len(buy_signals) for symbol in buy_signals}
        sell_weights = {symbol: weight / len(sell_signals) for symbol in sell_signals}

        weights = {**buy_weights, **{s: -weight for s in sell_weights}}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest