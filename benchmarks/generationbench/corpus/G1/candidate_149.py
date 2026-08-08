from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a significant breakout in one direction, prices often reverse and continue "
        "towards the opposite extreme. This strategy looks for stocks that have just "
        "broken out above their recent range and are now near their highest price within"
        "that range."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_points: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window * 2 + 1:
                continue

            low = min(values[:self._window])
            high = max(values[-self._window:])
            breakout_price = max(values[self._window:self._window * 2])

            if values[self._window] > low and breakout_price >= high:
                breakout_points.append(symbol)

        breakout_points = breakout_points[: self._top_n]
        if not breakout_points:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_points)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_points}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest