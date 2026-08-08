from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can lead to "
        " continuation of the trend. By identifying symbols that show a significant price move"
        " accompanied by increased volume, we aim to capture profitable trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Identify the top 5 symbols with significant price changes and volume increase
        moves_and_volumes = (
            history.group_by("symbol")
            .agg(
                pl.col("close").last().alias("latest_close"),
                (pl.col("volume") - pl.col("volume").shift(1)).abs()
                .gt(pl.col("volume").mean())
                .alias("is_volume_increased"),
            )
            .sort("symbol", descending=True)
            .select(
                "symbol",
                "latest_close",
                "is_volume_increased",
                (pl.col("close") / pl.col("open") - 1).abs()
                .shift(-1)
                .alias("price_move"),
            )
        )

        top_symbols = moves_and_volumes.filter(
            (pl.col("is_volume_increased")) & (pl.col("price_move").gt(0.05))
        )[:5]

        if top_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                row["symbol"]: weight for _, row in top_symbols.iter_rows()
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest