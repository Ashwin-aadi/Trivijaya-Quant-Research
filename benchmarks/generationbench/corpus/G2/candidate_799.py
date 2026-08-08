from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the price action of a stock is confined to a narrower "
        "range than its historical average. This can indicate potential buying or selling "
        "pressure, which can be exploited for profit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].drop_nulls().to_list()) < self._window:
                continue
            recent_highs = [float(v) for v in history.select(pl.col(symbol).max()).to_series().to_list()[0]]
            recent_lows = [float(v) for v in history.select(pl.col(symbol).min()).to_series().to_list()[0]]
            if len(recent_highs) >= self._window and len(recent_lows) >= self._window:
                high_range = max(high - low for high, low in zip(recent_highs, recent_lows))
                mean_range = sum(high - low for high, low in zip(recent_highs, recent_lows)) / self._window
                if high_range < 0.95 * mean_range:
                    picks.append(symbol)

        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().date()
    assert isinstance(newest, date)
    return newest