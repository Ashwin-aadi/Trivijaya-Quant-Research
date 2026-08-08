from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a strong breakout, the price often consolidates and then continues in "
        "the breakout direction. This strategy identifies such continuation patterns for "
        "potential buying opportunities."
    )

    def __init__(self, window: int = 20, threshold: float = 0.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].to_list()]
            if len(values) < self._window + 1:
                continue

            # Calculate the breakout condition
            latest_close, second_latest_close = values[-1], values[-2]
            if (latest_close - second_latest_close) / second_latest_close > self._threshold:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:5]  # Top N symbols for simplicity
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