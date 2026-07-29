from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMoves(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum and can suggest "
        "continued trend continuation. By identifying symbols where the volume on a price "
        "move is higher than recent average volumes, we aim to capture these high-momentum "
        "opportunities."
    )

    def __init__(self, window: int = 20, volume_threshold: float = 1.5) -> None:
        self._window = window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily price changes and volumes
        history = history.with_columns(
            (
                (pl.col("close") - pl.col("adj_close").shift(1))
                / pl.col("adj_close").shift(1)
                * 100.0
            ).alias("price_change"),
            (pl.col("volume")).alias("volume"),
        )

        # Group by symbol and calculate average volume over the window period
        avg_volume = history.group_by("symbol").agg(
            pl.col("volume").mean().alias("avg_volume")
        )

        # Filter out symbols with insufficient data or no significant price change
        filtered_history = (
            history.join(avg_volume, on="symbol", how="inner")
            .filter(pl.col("price_change") > 0)
            .with_columns(
                (pl.col("volume") / pl.col("avg_volume")).alias("volume_ratio")
            )
        )

        # Identify symbols with volume above average
        high_volume_symbols = filtered_history.filter(
            (pl.col("volume_ratio") >= self._volume_threshold) & (
                pl.col("price_change") > 0
            )
        ).select("symbol").to_series().to_list()

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