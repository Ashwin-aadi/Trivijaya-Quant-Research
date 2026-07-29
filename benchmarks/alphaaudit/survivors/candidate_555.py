from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screens help filter out less liquid stocks, reducing the risk of significant"
        " trading costs and ensuring that the portfolio can be easily traded without moving the market."
    )

    def __init__(self, liquidity_threshold: float = 10_000_000) -> None:
        self._threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = (
            history.filter(pl.col("volume") > self._threshold)
            .group_by("symbol")
            .agg(pl.count().alias("freq"))
            .sort("freq", descending=True)
            .select(["symbol"])
            .head(50)["symbol"]
            .to_list()
        )

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