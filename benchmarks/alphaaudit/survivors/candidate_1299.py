from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy screens for symbols with high liquidity before applying equal "
        "weighting among the top liquid assets. Higher liquidity reduces transaction costs "
        "and increases confidence in the ability to trade large volumes without affecting the asset's price."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        volume_df = history.select(
            pl.col("symbol"), (pl.col("volume").sum()).alias("total_volume")
        ).group_by("symbol").agg(pl.col("total_volume").mean().alias("avg_volume"))
        top_symbols = (
            volume_df.sort("avg_volume", descending=True)
            .head(10)
            .select("symbol")
            .to_series()
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest