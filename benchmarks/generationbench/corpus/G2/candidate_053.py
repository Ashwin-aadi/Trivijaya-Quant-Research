from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "A breakout from a range suggests that the underlying trend is changing direction. "
        "If this breakout is followed by an extended period of trading at or above (or below) "
        "the breakout level, it may indicate a continuation of the new trend. This strategy "
        "identifies such continuations for potential follow-up entry."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback:
                continue

            # Find the highest close during the lookback period
            max_close = max(values)
            breakout_date = next(d for d, c in zip(view.history()["session_date"], values) if c == max_close)

            # Check if the breakout level is sustained over the window period
            post_breakout_values = [float(v) for v in view.closes(lookback=self._window)[symbol].drop_nulls().to_list()]
            if all(post_breakout >= max_close for post_breakout in post_breakout_values):
                breakout_symbols.append(symbol)

        weight = 1.0 / len(breakout_symbols) if breakout_symbols else 0
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