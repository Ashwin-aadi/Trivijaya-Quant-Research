from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "If a stock breaks out of its recent range and then retraces part of that move, "
        "it is likely to continue in the breakout direction. This strategy identifies such "
        "breakouts and signals positions based on their potential for continuation."
    )

    def __init__(self, window: int = 20, retrace_fraction: float = 0.5) -> None:
        self._window = window
        self._retrace_fraction = retrace_fraction

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window * 2 + 1:
                continue
            high, low, close = max(values), min(values), values[-1]
            breakout_high = high - low
            retrace_low = close - (high - self._retrace_fraction * breakout_high)
            retrace_high = close + (high - self._retrace_fraction * breakout_high)

            if (
                close > high and
                any(v < retrace_high and v > retrace_low for v in values[:-1])
            ):
                picks.append(symbol)

        picks = list(set(picks))  # Remove duplicates
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