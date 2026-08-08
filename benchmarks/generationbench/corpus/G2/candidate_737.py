from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction they initially broke. By identifying stocks "
        "that have recently broken out and continue to move above their previous high, we can "
        "capitalize on this momentum."
    )

    def __init__(self, window: int = 20, continuation_days: int = 5) -> None:
        self._window = window
        self._continuation_days = continuation_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_days)

        if history.height < self._window + self._continuation_days:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol)
            close_values = [float(v) for v in df["adj_close"].to_list()]

            if len(close_values) < self._window + self._continuation_days:
                continue

            breakout_price = max(close_values[-self._window:])
            continuation_price = max(close_values[-self._continuation_days:])

            if close_values[-1] >= continuation_price > breakout_price:
                breakout_symbols.append(symbol)

        weights = {s: 1.0 / len(breakout_symbols) for s in breakout_symbols}
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