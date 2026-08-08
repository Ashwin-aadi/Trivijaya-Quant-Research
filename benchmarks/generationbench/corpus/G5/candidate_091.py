from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased market volatility and potential for breakout. "
        "High relative daily price range dispersion suggests that the market is consolidating, setting up "
        "for a significant move in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        ranges_df = (
            history
            .select(pl.col("symbol"), (pl.col("high") - pl.col("low")).alias("range"))
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        # Calculate relative dispersion of daily range for each symbol
        dispersions_df = (
            history
            .select("symbol", (pl.col("high") - pl.col("low")).alias("daily_range"))
            .join(ranges_df, on="symbol")
            .with_columns(
                (pl.col("daily_range") / pl.col("avg_range")).fill_null(0).sub(1).alias("dispersion")
            )
        )

        # Filter symbols based on relative dispersion
        high_dispersion_symbols = (
            dispersions_df.sort("dispersion", descending=True)
            .head(self._window // 5)["symbol"]
            .to_series()
            .to_list()
        )

        if not high_dispersion_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_dispersion_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_dispersion_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest