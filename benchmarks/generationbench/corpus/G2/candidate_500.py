from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest that a strong market sentiment is driving "
        "the price in one direction. By identifying these moves early, we can profit from the "
        "continuation of the trend."
    )

    def __init__(self, window: int = 50, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(["symbol", "session_date", pl.col("close").alias("price")])
        volume = history.select(["symbol", "session_date", pl.col("volume").alias("vol")])

        price_changes = (
            closes
            .group_by("symbol")
            .agg(
                (pl.col("price").shift(-1) - pl.col("price")).abs().alias("price_change"),
                pl.col("price").last().alias("latest_close"),
            )
            .sort("session_date", descending=True)
            .select(["symbol", "price_change", "latest_close"])
        )

        volume_changes = (
            volume
            .group_by("symbol")
            .agg(
                (pl.col("vol").shift(-1) - pl.col("vol")).abs().alias("volume_change"),
                pl.col("vol").last().alias("latest_vol"),
            )
            .sort("session_date", descending=True)
            .select(["symbol", "volume_change", "latest_vol"])
        )

        combined = (
            price_changes.join(volume_changes, on="symbol")
            .with_columns(
                (pl.col("price_change") / pl.col("latest_close")).alias("price_ratio"),
                (pl.col("volume_change") / pl.col("latest_vol")).alias("vol_ratio"),
            )
            .filter((pl.col("price_ratio") > 0) & (pl.col("vol_ratio") > self._threshold))
        )

        if combined.is_empty():
            return Signal(information_available_at=stamp, weights={})

        selected_symbols = [str(row["symbol"]) for row in combined.to_dicts()]
        weight_per_symbol = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight_per_symbol
                for symbol in selected_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest