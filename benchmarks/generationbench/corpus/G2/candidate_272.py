from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can lead to "
        "continued momentum. By identifying symbols with significant volume on a breakout or "
        "decline, we can capture potential continuation of the trend."
    )

    def __init__(self, window: int = 10, threshold_volume: float = 1000000) -> None:
        self._window = window
        self._threshold_volume = threshold_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify symbols with significant directional move and volume
        moves = (
            history.group_by("symbol")
            .agg(
                (pl.col("high").max() - pl.col("low").min()).alias("price_move"),
                (pl.col("volume")[-1] / pl.col("volume").mean().over("symbol")).alias("vol_change"),
            )
        )

        # Filter symbols with significant price move and volume increase
        filtered_moves = (
            moves.filter(
                (pl.col("price_move") > 0.05 * history["close"].max()) &
                (pl.col("vol_change").gt(1.5))
            )
            .sort("price_move", descending=True)
            .head(self._window)
        )

        # Ensure there are enough symbols to form a signal
        if filtered_moves.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks = [row["symbol"] for row in filtered_moves.to_dicts()]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest