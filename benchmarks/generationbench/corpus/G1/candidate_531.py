from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting involves giving more weight to "
        "assets with higher liquidity. This strategy aims to capture the "
        "benefits of diversification while favoring assets that are easier "
        "to trade without significantly impacting their price."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity")
            )
            .group_by("symbol")
            .agg(pl.col("liquidity").mean().alias("avg_liquidity"))
            .sort("avg_liquidity", descending=True)
            .to_dict(False)["symbol"]
        )

        if len(liquidity) < 5:
            return Signal(information_available_at=stamp, weights={})

        top_5 = liquidity[:5]
        weights = {s: 0.2 for s in top_5}
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