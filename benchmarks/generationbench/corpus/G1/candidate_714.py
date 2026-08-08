from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks that have performed better than the broader market "
        "can provide excess returns over time. This strategy focuses on relative strength."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        universe_mean = (
            closes
            .select(pl.col("adj_close").mean().alias("universe_mean"))
            .with_columns(
                (pl.col("adj_close") / pl.col("universe_mean") - 1).alias("relative_strength")
            )
            .sort("relative_strength", descending=True)
            .tail(self._window)["symbol"]
        ).to_list()

        if not universe_mean:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(universe_mean)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(universe_mean, [weight] * len(universe_mean))),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest