from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "This strategy exploits high trading volume changes to predict directional price moves. "
        "It identifies stocks with sudden volume spikes and confirms trends based on closing prices relative to their 50-day moving averages."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=60)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_change = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("volume") - pl.col("volume").shift(1)).alias("vol_change"),
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).alias("price_change"),
                pl.col("adj_close").mean().over([pl.arange()]).alias("50dma"),
            )
            .select(
                "symbol",
                (pl.col("vol_change") / pl.col("volume").mean()).alias("volume_change_percent"),
                (pl.col("price_change") > 0).cast(pl.int64).sum().alias("num_up_days"),
                (pl.col("price_change") < 0).cast(pl.int64).sum().alias("num_down_days"),
            )
            .collect()
        )

        if volume_change.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_change = (
            volume_change
            .with_columns(
                (pl.col("volume_change_percent") > pl.col("volume_change_percent").quantile(0.75)).alias("is_volume_spike"),
                (pl.col("num_up_days") - pl.col("num_down_days")).abs().rank(method="dense").desc().alias("trend_confidence")
            )
            .filter(
                (pl.col("symbol").is_in(view.symbols))
                & (pl.col("volume_change_percent").is_not_null())
                & (pl.col("50dma").is_not_null())
                & (pl.col("is_volume_spike"))
            )
            .sort("trend_confidence")
        )

        if volume_change.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        picks = [str(symbol) for symbol in volume_change.head(self._top_n)["symbol"].to_list()]
        weight = 1.0 / len(picks)
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