from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often lead to further price movement in the direction of the breakout. "
        "By identifying stocks that have recently broken out and continue trending, we can capture "
        "the momentum and potentially benefit from sustained price movements."
    )

    def __init__(self, window: int = 20, continuation_window: int = 10) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            # Check for a breakout condition
            if values[-1] > max(values[-self._window : -1]):
                # Check for continuation over the next few sessions
                continuation_values = [float(v) for v in closes[symbol][
                    pl.col("session_date").is_between(
                        stamp - self._continuation_window,
                        stamp - 1
                    )].drop_nulls().to_list()]
                if all(value > values[-2] for value in continuation_values):
                    breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))[:5]
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