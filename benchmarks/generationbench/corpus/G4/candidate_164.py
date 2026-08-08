from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy exploits the theme of 'volume-confirmed directional moves' in the Indian equity market by "
        "identifying trends with significant price movements and substantial trading volume. By focusing on "
        "high-volume confirmations, we aim to capture persistent price movements that are likely sustained due to strong "
        "market sentiment."
    )

    def __init__(self, price_threshold: float = 0.5, volume_multiplier: float = 2.0, top_n: int = 30) -> None:
        self._price_threshold = price_threshold
        self._volume_multiplier = volume_multiplier
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=21)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily price movement
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("price_movement")
            )
            .with_columns(
                ((pl.col("volume") / pl.col("volume").rolling_mean(window_size=20)) * self._volume_multiplier)
                .alias("volume_confirmation")
            )
            .select(["symbol", "session_date", "price_movement", "volume_confirmation"])
        )

        # Rank stocks based on the composite score
        history = (
            history.with_columns(
                (pl.col("price_movement") * 0.75 + pl.col("volume_confirmation") * 0.25).alias("composite_score")
            )
            .sort("composite_score", descending=True)
            .head(self._top_n)
        )

        weights = {row["symbol"]: 1.0 / self._top_n for row in history.to_dicts()}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest