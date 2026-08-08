from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong momentum and can indicate "
        "continuation of trends. By focusing on both price movement and volume, we aim to "
        "identify significant shifts in market sentiment."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_changes = (history["volume"].to_list()[1:] / history["volume"].shift(1).to_list()[:-1] - 1.0).alias("volume_change")
        price_changes = (history["close"] / history["close"].shift(1) - 1.0).alias("price_change")

        df = (
            history
                .with_columns(volume_changes, price_changes)
                .sort("session_date", descending=True)
                .select(pl.all().exclude(["volume_change", "price_change"]))
                .group_by("symbol")
                .agg([
                    (pl.col("close")[-1] / pl.col("close")[0]).alias("final_to_initial_ratio"),
                    pl.col("volume").sum().alias("total_volume"),
                    pl.col("volume_change").mean().alias("avg_volume_change"),
                    pl.col("price_change").mean().alias("avg_price_change")
                ])
        )

        symbols = df.filter(
            (pl.col("final_to_initial_ratio") > 1.05) & 
            (pl.col("total_volume") > 1.5 * df["total_volume"].quantile(0.75)) &
            (pl.col("avg_price_change").abs() > 0.02)
        ).select(pl.col("symbol")).to_list()

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
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