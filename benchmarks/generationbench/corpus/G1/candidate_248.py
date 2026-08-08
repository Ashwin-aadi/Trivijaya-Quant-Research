from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMoves(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong buying or selling pressure. "
        "This strategy captures such moves by identifying symbols with significant volume increases "
        "on days where the price also shows a substantial change in direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        moves = (
            history
            .filter((pl.col("close") > pl.col("open")) | (pl.col("high") == pl.col("close")))
            .group_by(["symbol"], maintain_order=True)
            .agg([
                (pl.col("volume").sum() / self._window).alias("total_volume"),
                ((pl.col("close") - pl.col("open")).abs()).alias("price_move")
            ])
        )

        if moves.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            moves
            .sort("total_volume", descending=True)
            .head(self._window // 4)
            .select(["symbol"])
        )["symbol"].to_list()

        signal_weights = {s: 1.0 for s in top_symbols}
        return Signal(information_available_at=stamp, weights=signal_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest