from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Price levels revert to a mean over time due to mean reversion theory. "
        "Prices that have deviated significantly from their historical mean are likely to move back towards it."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (closes.select(pl.col("adj_close").mean())["adj_close"][0]).round(decimals=2)

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            recent_close = adj_closes[-1]
            diff_from_mean = (recent_close - mean_close).round(decimals=2)
            if abs(diff_from_mean) > 0.5:
                weight = 2 / len(view.symbols) * (1 - abs(diff_from_mean))
                weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items() if w != 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest