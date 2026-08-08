from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Reversion to the mean suggests that assets which have deviated significantly from "
        "their historical price levels will eventually return. By identifying symbols that "
        "have had extreme price movements and betting against these extremes, we can take "
        "advantage of the tendency for prices to revert."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        means = [sum(closes[i : i + self._window]) / self._window for i in range(len(closes) - self._window + 1)]
        deviations = [(closes[i] - means[i]).abs() for i in range(self._window, len(closes))]
        
        symbols_with_extreme_deviations: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            latest_index = -1
            max_deviation = 0.0
            for i, d in enumerate(deviations):
                if d > max_deviation and len(values[i:i + self._window]) == self._window:
                    max_deviation = d
                    latest_index = i
            
            if latest_index != -1:
                symbols_with_extreme_deviations.append(symbol)

        weights: dict[str, float] = {symbol: 0.0 for symbol in view.symbols}
        if symbols_with_extreme_deviations:
            weight_per_symbol = -1.0 / len(symbols_with_extreme_deviations)
            for symbol in symbols_with_extreme_deviations:
                weights[symbol] = weight_per_symbol

        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest