from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets with higher returns relative to their peers in the same market index "
        "are expected to continue outperforming due to superior fundamentals or market sentiment. "
        "By selecting assets with a higher recent return compared to the broader market, we aim to capture this trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        nifty_returns = (closes["close"] - closes["adj_close"].shift(1)) / closes["adj_close"].shift(1)
        weights: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue
            daily_returns = [(adj_closes[i] - adj_closes[i-1]) / adj_closes[i-1] for i in range(1, len(adj_closes))]
            avg_return = sum(daily_returns[-self._window:]) / (len(daily_returns) % self._window or 1)
            if avg_return > nifty_returns.mean().to_list()[0]:
                weights[symbol] = 1.0

        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in weights.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest