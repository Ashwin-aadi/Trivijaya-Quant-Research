from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Seasonality effects suggest that stock prices exhibit predictable patterns based on "
        "calendar events or seasons. In the Indian market, certain months may consistently outperform others due to factors like festive periods, corporate actions, or macroeconomic trends."
    )

    def __init__(self, window: int = 30, lookback: int = 24) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Filter by recent history
        recent_closes = closes.sort("session_date", descending=True).head(self._window)

        # Calculate the mean and standard deviation for each symbol over the recent period
        symbols_mean_std = (
            recent_closes.group_by("symbol")
            .agg(
                (pl.col("adj_close").mean().alias("mean"),
                 pl.col("adj_close").std().alias("std"))
            )
            .with_columns(
                ((pl.col("adj_close") - pl.col("mean")) / pl.col("std")).abs().alias("z_score")
            )
        )

        # Identify symbols with the highest z-score (most deviated from mean)
        top_symbols = (
            symbols_mean_std.sort("z_score", descending=True).head(5)["symbol"].to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(top_symbols, [weight] * len(top_symbols))),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest