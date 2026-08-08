from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that after a significant price movement in one direction, "
        "the price will tend to revert to the mean. This strategy aims to capture such "
        "reversionary movements in the Indian market."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].to_list()]
            mean_close = sum(adj_closes[-self._window:]) / self._window
            latest_close = float(history[history["session_date"] == stamp]["adj_close"])
            if (latest_close - mean_close) / mean_close < -0.1:
                picks.append(symbol)

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in picks:
                continue
            weight = 1.0 / len(picks)
            weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest