from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy capitalizes on high-volume price movements to identify "
        "significant directional trends. High volume often precedes substantial "
        "price changes due to institutional or professional trading activity."
    )

    def __init__(self, window: int = 20, volume_threshold: float = 1.5, top_n: int = 10) -> None:
        self._window = window
        self._volume_threshold = volume_threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily high-low range
        history = (
            history
            .with_columns(
                (pl.col("high") - pl.col("low")).alias("range"),
                ((pl.col("close") - pl.col("adj_close").shift(1)) / pl.col("adj_close").shift(1) * 100).alias("price_change"),
                (pl.col("volume") - pl.col("volume").shift(1)).alias("volume_change")
            )
        )

        # Calculate average volume over the window
        avg_volume = history.select(pl.col("volume").mean().alias("avg_volume"))

        # Filter out symbols not in view.symbols
        filtered_history = history.filter(pl.col("symbol").is_in(view.symbols))
        
        # Group by symbol and calculate relevant metrics
        grouped = (
            filtered_history.join(avg_volume, on="session_date")
            .group_by("symbol")
            .agg([
                (pl.col("volume_change") > self._volume_threshold * pl.col("avg_volume")).alias("volume_spike"),
                (pl.col("price_change").abs() > 2.0).alias("price_momentum"),
                pl.col("close").mean().alias("avg_close")
            ])
        )

        # Rank candidates based on volume spike and price momentum
        ranked = (
            grouped
            .with_columns(
                ((pl.col("volume_spike") * 10) + (pl.col("price_momentum") * 5)).rank(method="dense", descending=True).alias("score")
            )
            .sort("score")
            .select(["symbol", "avg_close"])
        )

        if ranked.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [str(row["symbol"]) for row in ranked.head(self._top_n)]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest