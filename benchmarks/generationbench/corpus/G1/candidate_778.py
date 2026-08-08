from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks exhibit stronger performance during specific times of the year. "
        "By identifying these seasonal effects, we can allocate capital to perform better in those periods."
    )

    def __init__(self, window: int = 365, seasonality_period: int = 4) -> None:
        self._window = window
        self._seasonality_period = seasonality_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Determine the seasonal period
        seasonality_period = 4  # Quarterly effect for this example

        # Calculate the average returns by season
        seasons = [history.select(pl.col("session_date").dt.quarter())]
        avg_returns_by_season = {
            season: history.filter(
                pl.col("session_date").dt.quarter() == season
            ).select(pl.col("adj_close")).to_dict()[0][1:]
            for season in range(1, 5)
        }

        # Find the season with the highest average return
        best_season = max(avg_returns_by_season.keys(),
                          key=lambda s: sum(avg_returns_by_season[s]) / len(avg_returns_by_season[s]))
        
        symbols_in_best_season = view.symbols
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            if history.select(pl.col("session_date").dt.quarter() == best_season).filter(
                pl.col(symbol) > 0
            ).height == 0:
                symbols_in_best_season.remove(symbol)

        # Allocate weights to the selected symbols
        weight = 1.0 / len(symbols_in_best_season)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in symbols_in_best_season}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest