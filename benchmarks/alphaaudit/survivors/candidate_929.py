from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by further movement in the same direction. "
        "This strategy identifies symbols that have recently broken out and "
        "continues to hold them if they maintain momentum."
    )

    def __init__(self, window: int = 20, threshold: float = 0.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + 1)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            # Calculate the breakout close and compare it to the next session's price
            last_breakout_close = values[-2]
            current_price = values[-1]
            if (current_price - last_breakout_close) / last_breakout_close >= self._threshold:
                breakout_symbols.append(symbol)

        weights = {s: 1.0 for s in breakout_symbols}
        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest