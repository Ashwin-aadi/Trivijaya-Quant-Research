from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Higher liquidity stocks are often more efficiently priced and have lower trading "
        "costs. By equally weighting these stocks, we aim to capture the benefits of market "
        "efficiency while avoiding the potential for higher transaction costs associated with "
        "less liquid securities."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_closes = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume") / 100_000).alias("scaled_volume")
            )
            .group_by("symbol")
            .agg(pl.col("scaled_volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .head(self._window)
        )

        symbols = liquidity_screened_closes["symbol"].to_list()[:10]
        weight = 0.1 if len(symbols) > 0 else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest