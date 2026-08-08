from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Seasonal effects can arise from various economic and social factors. For instance, "
        "certain industries might experience higher trading volumes or prices during specific months."
    )

    def __init__(self, season_window: int = 12) -> None:
        self._season_window = season_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._season_window)
        if history.height < self._season_window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = {symbol: float(history[symbol][-1]) for symbol in symbols}
        season_dates = [date(year=year, month=month, day=1) for year in range(2020, 2024) for month in range(1, 13)]

        seasonal_factors = {}
        for date_index in range(len(season_dates)):
            current_date = season_dates[date_index]
            symbol_closes = [closes[symbol] for symbol in symbols if closes[symbol] > 0]
            avg_close = sum(symbol_closes) / len(symbol_closes)
            seasonal_factors[current_date] = (max(symbol_closes) - min(symbol_closes)) / avg_close

        top_seasons = sorted(seasonal_factors.items(), key=lambda x: x[1], reverse=True)[:5]

        weights = {}
        for season, factor in top_seasons:
            if season.month == stamp.month and season.year == stamp.year:
                symbol_closes = [closes[symbol] for symbol in symbols]
                weight = 1.0 / len(symbols)
                for symbol in symbols:
                    if closes[symbol] > 0:
                        weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest