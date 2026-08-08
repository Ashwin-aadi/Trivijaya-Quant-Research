from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Breakouts are often followed by a continuation of the trend. Identifying "
        "breakout symbols and their continuation can provide profitable opportunities."
    )

    def __init__(self, window: int = 20, lookback_period: int = 1) -> None:
        self._window = window
        self._lookback_period = lookback_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history_df = view.history(lookback=self._window + self._lookback_period)

        if history_df.height < self._window + self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history_df.columns:
                continue
            close_series = history_df[symbol].drop_nulls()
            if len(close_series) < self._window + self._lookback_period:
                continue

            last_close = float(close_series[-1])
            prev_max = max(close_series[self._window : -self._lookback_period])
            next_open = float(history_df[symbol][self._window - 1])

            if last_close > prev_max and next_open >= last_close:
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))[: self._lookback_period]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest