from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong buying or selling pressure. "
        "By identifying such moves and holding the directionally strongest stocks, we aim to "
        "capitalize on the momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if f"{s}_volume" in history.columns and all(history.select(pl.col(s).alias("symbol")).height > 0)]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the directional moves
        moves = (
            history.select(
                pl.col("symbol").alias("symbol"), "session_date", (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("directional_move")
            )
            .group_by("symbol")
            .agg((pl.col("directional_move").sum().abs()).alias("total_move"), (pl.col("volume").sum()).alias("total_volume"))
        )

        # Filter out negative directional moves
        filtered_moves = moves.filter(pl.col("directional_move") >= 0)

        # Rank by the sum of absolute directional moves and total volume
        ranked_moves = (
            filtered_moves.sort("total_move", descending=True)
            .sort("total_volume", descending=True)
            .select(pl.col("symbol"))
            .head(5)
        )

        if ranked_moves.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_moves)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_moves.to_list()[0]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest