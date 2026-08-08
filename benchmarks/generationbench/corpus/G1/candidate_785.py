from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After an initial breakout, the price often continues in the direction of the "
        "breakout. This strategy identifies such continuation patterns and allocates capital to"
        " these symbols."
    )

    def __init__(self, window: int = 20, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_series) < self._window + 1:
                continue

            breakout_price = max(close_series[-2:])
            continuation_price = min(close_series[-self._window :])
            if (
                close_series[-1] >= breakout_price * self._threshold
                and continuation_price > (breakout_price - history[symbol][-2]) / 2 + history[symbol][-2]
            ):
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))
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