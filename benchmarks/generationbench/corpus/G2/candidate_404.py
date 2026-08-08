from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can be a strong indicator of momentum. "
        "Large volume surges in the direction of recent price movement suggest that buying or selling pressure is building."
    )

    def __init__(self, window: int = 10, min_volume_threshold: float = 500_000) -> None:
        self._window = window
        self._min_volume_threshold = min_volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        latest_close = view.latest_close()

        # Calculate directional move and volume for each symbol
        def directional_move(row):
            close_diff = row['close'] - row['adj_close'].shift(1)
            if close_diff > 0:
                return 'up'
            elif close_diff < 0:
                return 'down'
            else:
                return None

        moves_df = history.with_columns(
            (pl.col("close") - pl.col("adj_close").shift(1)).alias("close_diff"),
            pl.when(pl.col("volume") > self._min_volume_threshold)
                 .then(directional_move(pl.extract_row()))
                 .otherwise(None).alias("direction")
        )

        # Filter out symbols with insufficient volume or no direction
        valid_symbols = moves_df.filter(
            (pl.col("volume") > self._min_volume_threshold) &
            (pl.col("direction").is_not_null())
        ).select(["symbol", "direction"]).collect().to_dict(False)

        if not valid_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected symbol
        weights = {sym: 1.0 / len(valid_symbols) for sym in valid_symbols['symbol']}
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