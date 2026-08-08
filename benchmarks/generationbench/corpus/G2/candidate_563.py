from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term momentum with long-term trend-following can provide a more "
        "robust signal. Short-term momentum captures recent strength, while long-term trends "
        "indicate sustained performance."
    )

    def __init__(self, short_window: int = 10, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._long_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        short_moments = (history["adj_close"] / history["adj_close"].shift(self._short_window) - 1.0).alias("short_moment")
        long_trends = (history["adj_close"] / history["adj_close"].shift(self._long_window) - 1.0).alias("long_trend")

        momentum_history = history.with_columns(short_moments, long_trends)
        if momentum_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        positive_short = (momentum_history["short_moment"] > 0).sum()
        negative_long = (momentum_history["long_trend"] < 0).sum()

        if positive_short == 0 or negative_long >= self._long_window:
            return Signal(information_available_at=stamp, weights={})

        picks = [symbol for symbol in view.symbols if symbol in momentum_history.columns]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest