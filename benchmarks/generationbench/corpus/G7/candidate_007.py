from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Dispersion20d(Strategy):
    rationale = (
        "The dispersion of daily High-Low ranges over a 20-day period can indicate "
        "whether the market is trending or range-bound. Higher dispersion suggests more "
        "volatility and less predictability, making it a potential opportunity for trading."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_low_ranges = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range"),
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("mean_range"))
        )

        top_symbols = (
            high_low_ranges.sort("mean_range", descending=True)
            .head(self._top_n)
            .select("symbol")
            .to_series()
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights=dict(zip(top_symbols, [weight] * len(top_symbols)))
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest