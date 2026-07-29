from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key factor in marketability and can indicate the ease with which an asset "
        "can be bought or sold without affecting its price. This strategy screens for high liquidity "
        "assets and equally weights them to create a diversified portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = (
            history.select(["symbol", "volume"])
            .filter(pl.col("volume").is_not_null())
            .group_by("symbol")
            .agg((pl.col("volume").mean()).alias("avg_volume"))
            .sort("avg_volume", descending=True)
            .head(10)["symbol"]
        ).to_list()

        if not high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_volume_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_volume_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest