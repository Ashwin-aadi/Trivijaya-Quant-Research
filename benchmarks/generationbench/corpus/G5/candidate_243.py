from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a measure of the ease with which an asset can be bought or sold without "
        "causing significant movement in its price. A screen based on liquidity ensures that only "
        "the most liquid stocks are selected, reducing transaction costs and execution risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume").sum() / 100_000).alias("total_volume")
            )
            .group_by("symbol")
            .agg(pl.col("total_volume").mean().alias("avg_volume"))
            .sort(by="avg_volume", descending=True)
        )

        symbols = liquidity_screened["symbol"].to_list()
        if len(symbols) < 10:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.1
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols[:10]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().item()
    assert isinstance(newest, date)
    return newest