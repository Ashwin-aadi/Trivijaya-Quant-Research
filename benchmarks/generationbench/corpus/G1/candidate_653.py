from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a strong breakout, the stock often continues in that direction. "
        "By identifying such stocks and holding them, we can benefit from this momentum."
    )

    def __init__(self, window: int = 20, continuation_window: int = 10) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)
        if history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._continuation_window:
                continue

            breakout_index = max(range(len(values)), key=lambda i: values[i])
            if breakout_index >= self._window and (
                (values[breakout_index] - min(values[:self._window])) / (max(values[:self._window]) - min(values[:self._window]))
                > 0.3
            ):
                continuation = [values[i + 1] for i in range(self._continuation_window) if i + 1 < len(values)]
                if all(value > values[breakout_index] for value in continuation):
                    breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:5]
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