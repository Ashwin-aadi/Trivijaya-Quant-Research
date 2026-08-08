from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Historical stock markets exhibit seasonal patterns where returns are higher or lower "
        "during specific times of the year. By identifying these patterns, we can construct a "
        "strategy that allocates capital towards symbols with historically strong performance at "
        "certain times."
    )

    def __init__(self, window: int = 10, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_performance = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the average return over the past `self._window` days
            avg_return = sum((values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))) / (len(values) - 1)
            symbol_performance[symbol] = avg_return

        # Sort symbols by their average return and pick the top N
        sorted_symbols = sorted(symbol_performance.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        picks = [s for s, _ in sorted_symbols]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest