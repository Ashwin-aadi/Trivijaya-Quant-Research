from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToMean(Strategy):
    rationale = (
        "This strategy capitalizes on the tendency of stock prices to revert to historical "
        "averages. It identifies stocks that have deviated significantly from their mean price "
        "levels and trades in the direction opposite to this deviation."
    )

    def __init__(self, window: int = 50, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_prices = (
            history.groupby("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean_price"))
                   .with_columns((pl.col("adj_close") - pl.col("mean_price")).abs().alias("deviation"))
                   .sort("deviation", descending=True)
                   .head(self._threshold * self._window)["symbol"]
        )

        weights = {s: 1.0 / len(mean_prices) for s in mean_prices}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest