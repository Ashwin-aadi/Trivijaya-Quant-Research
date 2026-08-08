from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data may show that certain stocks in the NIFTY 100 exhibit seasonality "
        "effects where their performance varies significantly during specific times of the year. "
        "For instance, some sectors might perform better in particular months due to seasonal trends."
    )

    def __init__(self, window: int = 365, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the mean return over different seasons and store them.
            seasonality_factors[symbol] = _calculate_seasonal_return(values, stamp)

        sorted_symbols = [k for k, v in sorted(seasonality_factors.items(), key=lambda item: -item[1])]
        picks = sorted_symbols[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

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


def _calculate_seasonal_return(values: list[float], stamp: date) -> float:
    # Determine the season for each session and compute returns.
    seasons = ["Q1", "Q2", "Q3", "Q4"]
    quarter_start_dates = [date(stamp.year, 1, 1), date(stamp.year, 4, 1),
                           date(stamp.year, 7, 1), date(stamp.year, 10, 1)]

    returns_by_season: dict[str, float] = {}
    for i in range(len(values)):
        session_date = values[i - len(values)].name
        quarter_index = next(i for i, d in enumerate(quarter_start_dates) if session_date >= d)
        season = seasons[quarter_index]

        # Calculate the return for this season.
        if season not in returns_by_season:
            returns_by_season[season] = (values[i] - values[0]) / values[0]
        else:
            returns_by_season[season] += (values[i] - values[0]) / values[0]

    # Calculate the average return for each season.
    for k, v in returns_by_season.items():
        returns_by_season[k] = v / len(values)

    # Identify the season with the highest return.
    best_season = max(returns_by_season, key=returns_by_season.get)
    return returns_by_season[best_season]