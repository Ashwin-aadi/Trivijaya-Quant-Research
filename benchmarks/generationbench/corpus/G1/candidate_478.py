from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit stronger performance during specific seasons "
        "of the year. By identifying these seasonal patterns, we can allocate capital to stocks that are expected "
        "to perform well at certain times."
    )

    def __init__(self, season: int = 3) -> None:
        self._season = season

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365 * 4)  # Look at the last 4 years of data
        if history.height < 1095:  # Ensure we have enough data
            return Signal(information_available_at=stamp, weights={})

        season_map = {3: "Summer", 6: "Monsoon", 9: "Winter", 12: "Post-Monsoon"}
        current_season = (view.as_of.month - 1) // 3 + 1
        target_season = (current_season + self._season) % 4

        season_changes = (
            history.filter(pl.col("session_date").dt.quarter() == target_season)
                  .group_by("symbol")
                  .agg((pl.col("adj_close") / pl.col("adj_close").shift(120).fill_null(1.0) - 1.0).alias("seasonal_return"))
        )
        
        if season_changes.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = season_changes.sort("seasonal_return", descending=True)["symbol"].to_list()[:5]
        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest