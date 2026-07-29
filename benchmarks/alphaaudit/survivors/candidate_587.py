from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy combines short-term and long-term momentum to identify "
        "stocks with strong trending behavior. Short-term momentum signals "
        "strength in the recent past while long-term momentum indicates "
        "a sustained upward or downward trend."
    )

    def __init__(self, short_window: int = 10, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._short_window, self._long_window))
        if closes.height < max(self._short_window, self._long_window):
            return Signal(information_available_at=stamp, weights={})

        short_moments: list[float] = []
        long_moments: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < max(self._short_window, self._long_window):
                continue

            short_change = (values[-1] - values[-self._short_window]) / abs(values[-self._short_window])
            long_change = (values[-1] - values[-self._long_window]) / abs(values[-self._long_window])

            short_moments.append(short_change)
            long_moments.append(long_change)

        short_threshold = 0.2
        long_threshold = 0.5

        symbols_with_short_positive = [symbol for symbol, moment in zip(view.symbols, short_moments) if moment >= short_threshold]
        symbols_with_long_positive = [symbol for symbol, moment in zip(view.symbols, long_moments) if moment >= long_threshold]

        intersection = list(set(symbols_with_short_positive).intersection(symbols_with_long_positive))
        if not intersection:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(intersection)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in intersection}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest