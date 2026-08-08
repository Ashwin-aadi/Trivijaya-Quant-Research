from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum and can lead to "
        "potentially profitable opportunities. By identifying symbols that show a significant"
        " price movement accompanied by increased trading volume, we aim to capture such trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate price change and volume change
        moves = (
            history.select(
                pl.col("symbol"),
                (pl.col("close") - pl.col("adj_close").shift(self._window)).alias("price_change"),
                (pl.col("volume") / pl.col("volume").mean()).rank(method="dense", descending=True).alias("vol_rank")
            )
            .filter(pl.col("price_change") > 0)
        )

        # Filter for both significant price change and high volume
        filtered_moves = (
            moves.filter(
                (pl.col("price_change") / history.select(pl.col("close")).row(0)[1] >= 0.1) & 
                (pl.col("vol_rank") <= 5)
            )
            .group_by("symbol", maintain_order=True)
            .agg(
                (pl.col("price_change").max() / history.select(pl.col("close")).row(0)[1]).alias("rel_move"),
                pl.col("vol_rank").min().alias("vol_rank")
            )
        )

        if filtered_moves.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            filtered_moves.sort(
                "rel_move", descending=True
            ).select(pl.col("symbol")).head(5).to_list()[0]
        )

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