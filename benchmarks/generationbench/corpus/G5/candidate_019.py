from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price action is consolidating, which can precede "
        "a breakout or a trend reversal. By identifying symbols with reduced volatility and "
        "recently closing near their daily highs or lows, we can prepare for potential significant "
        "moves."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        history = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("daily_range")
            )
            .group_by("symbol")
            .agg(
                (pl.col("daily_range").mean().alias("avg_range")),
                (pl.col("daily_range").std().alias("range_std")),
                (
                    (pl.col("close") >= pl.col("high").shift(-1)).sum()
                    / self._window
                    * 100.0
                ).alias("percent_close_above_high"),
                (
                    (pl.col("close") <= pl.col("low").shift(-1)).sum()
                    / self._window
                    * 100.0
                ).alias("percent_close_below_low"),
            )
        )

        # Identify symbols with reduced volatility and recent closing near daily highs or lows
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or history.height < 1:
                continue
            avg_range, range_std = float(history[symbol]["avg_range"][0]), float(
                history[symbol]["range_std"][0]
            )
            percent_close_above_high, percent_close_below_low = (
                float(history[symbol]["percent_close_above_high"][0]),
                float(history[symbol]["percent_close_below_low"][0]),
            )

            # Flags symbols with a small average daily range and a high percentage of closes near highs or lows
            if avg_range <= 2 * range_std and (percent_close_above_high > 85.0 or percent_close_below_low > 85.0):
                picks.append(symbol)

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