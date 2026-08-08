from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "After a significant price move in one direction, a confirmation through volume "
        "suggests that the trend is likely to continue, providing an opportunity for "
        "profits."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        price_changes = (
            history.sort("session_date")
            .select(
                pl.col("symbol"),
                (pl.col("close") - pl.col("adj_close").shift(1)).alias("price_change"),
            )
            .group_by("symbol")
            .agg(pl.sum("price_change").alias("total_price_change"))
        )

        strong_moves = price_changes.filter(
            pl.col("total_price_change").abs() > 0.05
        ).select(pl.col("symbol")).to_list()[0]

        if not strong_moves:
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed = (
            view.history()
            .filter(pl.col("symbol").is_in(strong_moves))
            .group_by("symbol")
            .agg(
                pl.sum("volume").alias("total_volume"),
                (pl.col("close") - pl.col("adj_close").shift(1)).alias("price_change")
            )
        )

        high_volume_symbols = volume_confirmed.filter(
            pl.col("total_volume") > 100_000
        ).select(pl.col("symbol")).to_list()[0]

        if not high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_volume_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_volume_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest