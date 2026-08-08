from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of low price volatility (range compression), stocks are likely to "
        "break out of their ranges. This strategy identifies symbols with reduced daily "
        "price dispersion and bets on these."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_ranges = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).mean().alias("range_mean"),
                pl.col("close").stddev().mean().alias("volatility_mean")
            )
        )

        # Filter out symbols with NaN volatility or range mean
        filtered_ranges = mean_ranges.filter(pl.all-columns().is_not_null())

        if filtered_ranges.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sorted_ranges = filtered_ranges.sort("range_mean", descending=True)
        top_symbols = [row["symbol"] for row in sorted_ranges.rows()][:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest