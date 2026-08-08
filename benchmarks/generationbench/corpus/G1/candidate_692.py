from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "We look for significant price movements that are accompanied by increased volume to filter out noise and identify genuine trends."
    )

    def __init__(self, window: int = 20, min_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._min_volume_ratio = min_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_changes = (
            history.with_columns(
                (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("vol_change")
            )
            .group_by("symbol")
            .agg(
                (pl.col("close").last() / pl.col("close").first()).alias("price_move"),
                (pl.col("adj_close").sum()).alias("total_adj_close"),
                (pl.col("volume").sum()).alias("total_volume"),
            )
        )

        if volume_changes.height < 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            row = volume_changes.filter(pl.col("symbol") == symbol).row(0)
            price_move = float(row[0])
            vol_change = float(row[1])

            if (
                abs(price_move) >= 0.05
                and vol_change > self._min_volume_ratio
                or -vol_change > self._min_volume_ratio
            ):
                breakout_symbols.append(symbol)

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest