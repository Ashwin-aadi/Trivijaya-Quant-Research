from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakouts occur when a security breaks out of its recent range but "
        "fails to close outside it. This suggests that buying interest remains strong and the "
        "move is likely to continue in the direction of the breakout."
    )

    def __init__(self, window: int = 20, lookback: int = 5) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            recent_high = max(adj_closes[-self._window:])
            recent_low = min(adj_closes[-self._window:])
            breakout_high = max(adj_closes)
            if (breakout_high > recent_high and
                    adj_closes[-1] < recent_high):
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[: self._lookback]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest