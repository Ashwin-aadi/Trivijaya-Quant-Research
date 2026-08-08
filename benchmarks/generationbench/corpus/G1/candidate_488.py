from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "A directional move in price accompanied by an increase in volume suggests a "
        "stronger momentum and greater conviction from market participants. This strategy "
        "identifies such moves to capitalize on the increased likelihood of sustained trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .with_columns(pl.col("symbol").alias("symbol_id"))
        )

        # Filter out rows where the return is zero or null
        history = history.filter((pl.col("return") != 0) & (pl.col("return").is_not_null()))

        # Calculate daily volume change
        history = (
            history.with_columns(
                ((pl.col("volume") - pl.col("volume").shift(1)) / pl.col("volume").shift(1) * 100).alias("vol_change")
            )
            .sort("session_date", descending=False)
        )

        # Identify breakout days
        breakout_days = history.select(
            [
                pl.col("symbol_id"),
                (pl.col("return") >= 0.05).cast(pl.int64) + (pl.col("return") <= -0.05).cast(pl.int64),
                (pl.col("vol_change") > 20).alias("high_vol"),
            ]
        ).filter((pl.col("return") >= 0.05) | (pl.col("return") <= -0.05)).sort("session_date", descending=False)

        # Get the latest breakout day
        last_breakout = breakout_days.sort("session_date", descending=True).rows(1)
        if not last_breakout:
            return Signal(information_available_at=stamp, weights={})

        symbol_id = last_breakout[0][0]
        high_vol = bool(last_breakout[0][2])

        symbols = history.filter(pl.col("symbol_id") == symbol_id)["symbol"].to_list()
        weight = 1.0 / len(symbols) if symbols else 0

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols if high_vol},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest