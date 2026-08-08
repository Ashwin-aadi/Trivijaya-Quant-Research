from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a strong price breakout, the continuation of this trend is often profitable. "
        "This strategy identifies symbols that have recently broken out and are likely to continue their momentum."
    )

    def __init__(self, window: int = 20, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            # Calculate the price change from the breakout point to the most recent close
            open_price = values[0]
            close_price = values[-1]
            high_price = max(values)
            low_price = min(values)

            # Check for a valid breakout condition
            if close_price >= high_price and (close_price - open_price) / open_price >= self._threshold:
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))  # Remove duplicates
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest