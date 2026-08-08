from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy exploits volume-confirmed directional moves by identifying significant price "
        "changes accompanied by high trading volumes. High-volume days often indicate strong investor "
        "sentiment and reliable trend confirmations."
    )

    def __init__(self, lookback: int = 30, top_n: int = 5) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.height < self._lookback * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily price change percentage
        history = (
            history.with_columns(
                ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias("price_change")
            )
            .group_by(["symbol", "session_date"])
            .agg(pl.col("price_change").mean().alias("avg_price_change"))
        )

        # Calculate moving average of volume
        history = (
            history.with_columns(
                (pl.col("volume") - pl.col("volume").shift(1)) / 30 * 100.0.alias("vol_change")
            )
            .group_by(["symbol", "session_date"])
            .agg(pl.col("vol_change").mean().alias("avg_vol_change"))
        )

        # Compute combined signal score
        history = (
            history.with_columns(
                (pl.col("avg_price_change") * pl.col("avg_vol_change")).alias("combined_signal")
            )
        )

        # Rank by combined signal and select top symbols
        history = history.sort("combined_signal", descending=True).select(
            [pl.col("symbol"), "session_date", "combined_signal"]
        ).head(self._top_n)

        if history.height == 0:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row["symbol"] for row in history]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={p: weight for p in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest