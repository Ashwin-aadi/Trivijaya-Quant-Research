from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy exploits significant volume surges that precede substantial price changes, "
        "capitalizing on the momentum created by large-volume trades. It aims to identify stocks with "
        "increased trading activity and corresponding price movements to enter long or short positions."
    )

    def __init__(self, lookback: int = 30, threshold: float = 1.2) -> None:
        self._lookback = lookback
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute volume and price change
        history = (
            history
            .with_columns(
                (pl.col("volume") / pl.col("volume").shift(self._lookback) - 1.0).alias("volume_ratio"),
                ((pl.col("adj_close") - pl.col("adj_close").shift(1)) / pl.col("adj_close").shift(1) * 100).alias("price_change")
            )
            .filter(
                (pl.col("volume_ratio") > self._threshold)
                & (pl.col("price_change") >= 1.0)
            )
        )

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Rank by the magnitude of volume increase and price change
        picks = (
            history
            .sort("volume_ratio", descending=True)
            .group_by("symbol")
            .agg([
                (pl.col("adj_close").mean()).alias("avg_price"),
                (pl.col("volume")).sum().alias("total_volume"),
                (pl.col("price_change").max()).alias("max_price_change")
            ])
            .sort("max_price_change", descending=True)
            .head(30)["symbol"]
            .to_list()
        )

        # Assign equal weights
        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest