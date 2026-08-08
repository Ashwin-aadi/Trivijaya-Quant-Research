from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies stocks that have recently broken out of their previous price ranges and are continuing to trend in the breakout direction. It uses daily OHLC prices to filter for strong continuation patterns with increased volume."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history(lookback=self._window)
            recent_closes = [float(v) for v in closes[symbol].to_list()]
            recent_history = [float(v["adj_close"]) for v in history.to_dicts()]

            # Check for breakout and continuation pattern with increased volume
            is_breakout_high = all(
                recent_closes[i] > max(recent_closes[:i])
                for i in range(self._window, len(recent_closes))
            )
            is_breakout_low = all(
                recent_closes[i] < min(recent_closes[:i])
                for i in range(self._window, len(recent_closes))
            )

            if not (is_breakout_high or is_breakout_low):
                continue

            # Check for volume increase over the last 5 days
            avg_volume = history.select(pl.col("volume").mean()).collect().height[0]
            recent_volumes = [float(v["volume"]) for v in history.to_dicts()[-5:]]
            if all(volume > avg_volume * 1.2 for volume in recent_volumes):
                signals[symbol] = 1.0 / self._top_n

        return Signal(information_available_at=stamp, weights={**signals})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest