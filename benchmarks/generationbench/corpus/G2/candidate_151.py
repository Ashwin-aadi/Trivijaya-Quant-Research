from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the price action is more confined within a smaller "
        "range than expected. This can indicate increased market uncertainty or a potential "
        "upcoming breakout. We seek symbols where recent range compression has been highest."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("adj_close").shift(-1) - pl.col("adj_open")).abs().alias("close_range"),
            )
            .with_columns(
                (pl.col("range") / history["open"].first()).alias("relative_range"),
                (pl.col("close_range") / history["open"].first()).alias("relative_close_range"),
            )
            .group_by("symbol")
            .agg(
                pl.col("relative_range").mean().alias("avg_relative_range"),
                pl.col("relative_close_range").max().alias("max_relative_close_range"),
            )
        )

        if range_compression.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [
            row[0]
            for row in range_compression.sort(
                "avg_relative_range", descending=False
            ).select("symbol").to_numpy()
        ][:5]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest