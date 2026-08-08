from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks that deviate "
        "significantly from their rolling mean. It aims to capture reversions towards historical "
        "averages driven by temporary market dynamics."
    )

    def __init__(self, window: int = 20, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:  # Ensure enough data points
                continue

            mean = sum(values) / self._window
            std_dev = (sum((v - mean) ** 2 for v in values) / self._window) ** 0.5
            last_close = values[-1]

            if abs(last_close - mean) > self._threshold * std_dev:
                symbols.append(symbol)

        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(
            information_available_at=stamp, weights={**weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest