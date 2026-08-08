from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when a security's price deviates from its mean level and "
        "eventually returns to it. This is based on the statistical property that extreme "
        "values are likely to revert back towards their historical average."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes
            .group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("mean_close"))
            .with_columns(
                (pl.col("adj_close") - pl.col("mean_close")).abs().alias("deviation"),
                ((pl.col("adj_close") / pl.col("mean_close")) - 1.0).alias("z_score")
            )
        )

        # Select symbols with the highest deviation
        picks: list[str] = mean_close.sort("deviation", descending=True)["symbol"].to_list()[:5]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest