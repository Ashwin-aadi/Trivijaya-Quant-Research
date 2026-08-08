from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a significant breakout, the market often continues in the same direction. "
        "This strategy identifies such breakouts and holds for a continuation period."
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
            if symbol not in history.columns or len(history[symbol].drop_nulls().to_list()) < self._window + self._continuation_window:
                continue

            open_vals = [float(v) for v in history[f"{symbol}.open"].drop_nulls().to_list()]
            close_vals = [float(v) for v in history[f"{symbol}.close"].drop_nulls().to_list()]

            if (max(close_vals[:self._window]) < min(open_vals[-self._continuation_window:]) or
                    max(close_vals[-self._continuation_window:]) < min(open_vals[:self._window])):
                breakout_symbols.append(symbol)

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