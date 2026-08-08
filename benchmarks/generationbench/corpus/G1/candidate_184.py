from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reversion occurs when a stock's price returns to its mean after moving "
        "away from it. Using a trailing reference point helps identify overbought or oversold conditions."
    )

    def __init__(self, window: int = 50, k: float = 2) -> None:
        self._window = window
        self._k = k

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(view.symbols) < 2:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(
            pl.col("adj_close").mean().alias("mean")
        ).to_dict(True)[0]["mean"]
        std_dev_close = history.select(
            pl.col("adj_close").stddev().alias("std")
        ).to_dict(True)[0]["std"]

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            recent_closes = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()
            if len(recent_closes) < self._window:
                continue
            z_score = (view.latest_close()[symbol] - mean_close) / std_dev_close
            if abs(z_score) > self._k:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        weights = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest