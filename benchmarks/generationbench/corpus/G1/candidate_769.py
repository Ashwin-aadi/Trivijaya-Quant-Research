from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies look for stocks that have deviated significantly from "
        "their historical average. If a stock's price has dropped sharply relative to its mean,"
        " it is expected to revert back towards the mean in the short term."
    )

    def __init__(self, window: int = 10, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_prices: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            mean_price = sum(values) / self._window
            mean_prices[symbol] = mean_price

        signals: dict[str, float] = {}
        for symbol, close in view.latest_close().items():
            if symbol not in mean_prices:
                continue
            if (close - mean_prices[symbol]) / mean_prices[symbol] > self._threshold:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        weight_per_symbol = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp, weights=weight_per_symbol
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest