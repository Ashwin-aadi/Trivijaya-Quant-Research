from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakouts occur when a stock that has already broken out of its range "
        "continues to rise. This can be interpreted as strong conviction among buyers who "
        "entered at the breakout and are now looking for further appreciation."
    )

    def __init__(self, window: int = 20, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)
        if history.is_empty() or history.height < self._window + 20:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window + 20:
                continue

            breakout_price = max(prices[-self._window:])
            continuation_price = prices[-1]

            if continuation_price > breakout_price * self._threshold:
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