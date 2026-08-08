from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term and long-term momentum signals aims to capture both the current "
        "trend strength and its persistence over time. Short-term momentum identifies recent "
        "price action, while long-term momentum indicates sustained performance."
    )

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_window + self._short_window - 1)
        if closes.height < self._long_window + self._short_window - 1:
            return Signal(information_available_at=stamp, weights={})

        short_moments: list[str] = []
        long_moments: list[str] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            short_values = [float(v) for v in closes[symbol].drop_nulls().to_list()[-self._short_window :]]
            long_values = [float(v) for v in closes[symbol].drop_nulls().to_list()[-self._long_window :]]

            if len(short_values) < self._short_window or len(long_values) < self._long_window:
                continue

            short_momentum = sum(short_values) / self._short_window
            long_momentum = sum(long_values) / self._long_window

            if short_momentum > 0 and long_momentum > 0:
                short_moments.append(symbol)
                long_moments.append(symbol)

        combined_symbols = list(set(short_moments).intersection(long_moments))
        if not combined_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in combined_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest