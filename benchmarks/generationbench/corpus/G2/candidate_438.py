from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Securities with higher returns relative to their peers in the NIFTY 100 index are "
        "expected to outperform over the medium term. This strategy exploits this idea by "
        "identifying securities that have outperformed their contemporaries."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        returns = (closes / closes.shift(1) - 1.0).to_series().alias("returns")

        mean_returns = history.select(
            pl.col("symbol").alias("symbol"),
            returns.mean().alias("mean_returns")
        ).sort("mean_returns", descending=True)

        top_n_symbols = mean_returns["symbol"].to_list()[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest