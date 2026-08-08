from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a breakout, a price retracement can indicate buying or selling pressure. "
        "A breakout followed by a price movement in the same direction for at least 3 days "
        "suggests continuation of the trend."
    )

    def __init__(self, lookback_window: int = 20, continuation_window: int = 3) -> None:
        self._lookback_window = lookback_window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbol = None
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            price_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(price_series) < self._lookback_window + 1:
                continue

            # Identify the breakout
            if price_series[-1] > max(price_series):
                breakout_symbol = symbol
                break

        if not breakout_symbol:
            return Signal(information_available_at=stamp, weights={})

        # Check for continuation
        after_breakout = [float(v) for v in closes[breakout_symbol].to_list()[-(self._continuation_window + 1):]]
        upward_trend = all(after_breakout[i] > after_breakout[i - 1] for i in range(1, self._continuation_window + 1))

        if not upward_trend:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(view.symbols)
        return Signal(
            information_available_at=stamp,
            weights={breakout_symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest