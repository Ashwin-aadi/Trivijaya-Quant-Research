from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to continue. By identifying symbols that "
        "show a significant price move in one direction accompanied by increased volume, we can "
        "capture momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        price_changes = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("price_change")
        volume_changes = (history["volume"] / history["volume"].shift(1) - 1.0).alias("volume_change")

        df = (
            history
                .with_columns(price_changes, volume_changes)
                .group_by("symbol")
                .agg([
                    pl.col("adj_close").last().alias("latest_price"),
                    (pl.col("price_change") > 0 & pl.col("volume_change") > 0).sum().alias("up_volume"),
                    (pl.col("price_change") < 0 & pl.col("volume_change") < 0).sum().alias("down_volume")
                ])
        )

        df = (
            df
                .filter(
                    (pl.col("up_volume") > self._window / 2) |
                    (pl.col("down_volume") > self._window / 2)
                )
                .sort("latest_price", descending=True)
                .head(5)
        )

        if df.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(df.height)
        weights = {row["symbol"]: float(weight) for row in df.rows()}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest