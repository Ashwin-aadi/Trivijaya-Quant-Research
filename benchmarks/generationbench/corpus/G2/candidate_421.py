from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that security prices and rental costs of assets will eventually "
        "return to the mean or average. In a short horizon, if an asset has deviated significantly "
        "from its historical price range, it is likely to revert back towards that mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean"))
                   .with_columns(
                       (pl.col("adj_close") - pl.col("mean")).abs().rank(method="dense",
                                                                          descending=True)
                   )
        )

        # Filter symbols with the smallest absolute deviations from mean
        filtered_symbols = (
            mean_close.sort("mean", descending=False)
                      .select(["symbol"])
                      .head(self._window)[0]
                      .to_list()
        )

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp, weights=dict(zip(filtered_symbols, [weight] * len(filtered_symbols)))
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest