from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After identifying a breakout, this strategy looks for stocks that continue to "
        "move in the direction of the breakout over the next period. This can indicate strong "
        "momentum and continuation of the trend."
    )

    def __init__(self, window_breakout: int = 20, window_continuation: int = 10) -> None:
        self._window_breakout = window_breakout
        self._window_continuation = window_continuation

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_breakout + self._window_continuation)
        if history.height < self._window_breakout + self._window_continuation:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            closes = [float(v) for v in history.filter(pl.col("symbol") == symbol).select("adj_close").to_series().drop_nulls()]
            if len(closes) < self._window_breakout + self._window_continuation:
                continue

            breakout_price = closes[-self._window_breakout]
            breakout_high = max(closes[-self._window_breakout:])
            continuation_high = max(closes[:-self._window_breakout]) if not all(v == breakout_price for v in closes[:self._window_breakout]) else None
            if continuation_high and continuation_high > breakout_high:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:5]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest