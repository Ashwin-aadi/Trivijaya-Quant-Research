from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a strong breakout, the continuation of that movement is often reliable. "
        "This strategy identifies symbols where the price continues to move in the direction"
        " of the recent breakout for a certain number of days."
    )

    def __init__(self, window_breakout: int = 20, window_continuation: int = 5) -> None:
        self._window_breakout = window_breakout
        self._window_continuation = window_continuation

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_breakout + self._window_continuation)

        if history.is_empty() or history.height < self._window_breakout + self._window_continuation:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = history[symbol].drop_nulls().to_list()
            if len(data) < self._window_breakout + self._window_continuation:
                continue

            # Calculate breakout direction
            breakout_direction = (data[-1] - data[0]) / data[0]
            if abs(breakout_direction) > 0.05:  # Consider breakouts with at least 5% move
                # Check continuation for the next window period
                close_price = [float(v) for v in data[-self._window_continuation:]]
                if breakout_direction * (close_price[-1] - close_price[0]) / close_price[0] > 0.02:
                    breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))[: self._window_continuation]
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