from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data often shows that certain stocks exhibit higher returns during specific times "
        "of the year due to seasonal effects. By identifying these patterns, we can allocate capital towards "
        "stocks expected to perform well in a given season."
    )

    def __init__(self, seasons: int = 4) -> None:
        self._seasons = seasons

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=365 * (self._seasons - 1))
        if closes.height < 365 * (self._seasons - 1):
            return Signal(information_available_at=stamp, weights={})

        # Define the seasons based on quarters
        seasonal_groups = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < 365 * (self._seasons - 1):
                continue

            # Calculate the mean close price for each season
            seasonal_data = [
                values[i : i + 90] for i in range(0, len(values), 90)
            ]  # Assuming a quarter is roughly 3 months or 90 days
            means = [sum(group) / len(group) for group in seasonal_data]

            # Identify the season with the highest mean close price
            max_mean_season = max(zip(means, range(len(seasonal_data))))
            best_season_index = max_mean_season[1]
            if best_season_index == 0:
                seasonal_groups.append(symbol)

        weight = 1.0 / len(seasonal_groups)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in seasonal_groups},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest