from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "A breakout followed by a significant retracement but still above the breakout level "
        "indicates strong demand and buying pressure. This strategy captures such opportunities."
    )

    def __init__(self, window: int = 20, breakout_window: int = 10) -> None:
        self._window = window
        self._breakout_window = breakout_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Check for breakout
            max_price = max(values[-self._breakout_window:])
            breakout_day = values.index(max_price)
            breakout_price = max_price
            low_after_breakout = min(values[breakout_day + 1 :])

            if values[-1] < low_after_breakout:
                continue

            # Ensure the price is above the breakout level but has retraced significantly
            if low_after_breakout <= (breakout_price * 0.8):
                picks.append(symbol)

        picks = picks[:5]
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