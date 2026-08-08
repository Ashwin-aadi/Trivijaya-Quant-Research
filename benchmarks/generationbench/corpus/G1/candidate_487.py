from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression is a period where the daily high and low are unusually close, "
        "indicating reduced volatility. In such periods, it can be an opportune time to enter "
        "positions in both directions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = [s for s in closes.columns if s not in ["session_date"]]
        
        means = history.groupby("symbol").agg(
            pl.col("high").min().alias("low_min"),
            pl.col("low").max().alias("high_max"),
            (pl.col("high") - pl.col("low")).mean().alias("range_mean")
        ).sort("range_mean", descending=True)

        if means.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [s for s in symbols if s in means.columns and means[means["symbol"] == s].shape[0] > 0]
        weight = 1.0 / len(top_symbols)

        weights = {s: weight for s in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest