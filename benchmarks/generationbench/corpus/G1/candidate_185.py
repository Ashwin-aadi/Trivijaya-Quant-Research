from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a breakout from a previous range, if the stock continues to move in the "
        "direction of the breakout, it may indicate sustained momentum and an opportunity for "
        "profit. This strategy identifies such continuation patterns."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].to_list()]
            breakout_price = max(close_values[:-self._window])
            current_close = close_values[-1]
            price_movement = (current_close - breakout_price) / breakout_price

            if price_movement > self._threshold:
                picks.append(symbol)

        picks = picks[:5]  # Top 5 symbols
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