from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy seeks to capitalize on trending markets by using volatility "
        "to scale into positions. During periods of low volatility, larger positions are taken, "
        "and during high volatility, positions are reduced or exited."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            closes.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").std().over(pl.col("session_date").rank(method="dense", descending=True).lt(self._vol_window)).alias("volatility"))
            )
            .collect()["volatility"]
            .to_list()
        )

        if all(v is None for v in volatility):
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s, v in zip(view.symbols, volatility) if v is not None]
        vol_weights = [(1 / (v + 0.01)) for v in volatility if v is not None]

        weight_sum = sum(vol_weights)
        if weight_sum == 0:
            return Signal(information_available_at=stamp, weights={})

        weights = {s: w / weight_sum for s, w in zip(symbols, vol_weights)}
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest