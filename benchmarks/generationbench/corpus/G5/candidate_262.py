from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit strong seasonal trends. "
        "Historical data suggests that certain months of the year consistently see higher returns for specific sectors or companies. "
        "This strategy aims to capitalize on these predictable patterns."
    )

    def __init__(self, season: str = "December", threshold: float = 0.1) -> None:
        self._season = season
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365 * 2)
        if history.height < 365 * 2:
            return Signal(information_available_at=stamp, weights={})

        season_dates = [
            date(year, self._season.month, self._season.day) + timedelta(days=-14),  # Adjust for week before
            date(year, self._season.month, self._season.day),
            date(year, self._season.month, self._season.day) + timedelta(days=14)
        ]
        
        seasonal_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            closes = [float(v) for v in history[history["symbol"] == symbol]["adj_close"].to_list()]
            season_closes = [closes[i - 1] for i, d in enumerate(history["session_date"]) if d in season_dates]
            non_season_closes = [closes[i - 1] for i, d in enumerate(history["session_date"]) if d not in season_dates]
            
            if len(season_closes) < 3 or len(non_season_closes) < 3:
                continue
            
            seasonal_return = (max(season_closes) / min(season_closes)) - 1.0
            non_seasonal_return = (max(non_season_closes) / min(non_season_closes)) - 1.0
            
            if abs(seasonal_return - non_seasonal_return) >= self._threshold:
                seasonal_returns[symbol] = seasonal_return

        picks: list[str] = []
        for symbol, return_ in seasonal_returns.items():
            if return_ > self._threshold:
                picks.append(symbol)

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