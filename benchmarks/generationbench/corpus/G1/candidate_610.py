from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Historical data suggests that certain stocks exhibit higher returns during specific times "
        "of the year due to seasonal effects or calendar anomalies. By identifying such patterns, we can "
        "anticipate favorable market conditions and allocate capital accordingly."
    )

    def __init__(self, window: int = 20, seasons: list[str] = ["Q1", "Q2"]) -> None:
        self._window = window
        self._seasons = seasons

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(closes.columns) != 101:
            return Signal(information_available_at=stamp, weights={})

        season_map = {date(2024, month, day).timetuple().tm_yday: season
                      for season in self._seasons
                      for month in range(3, 7) if (month == 3 and day >= 16) or (month != 3 and day <= 30)}
        season_close_prices = {symbol: [] for symbol in view.symbols}

        for i in range(self._window):
            session_date = stamp - date.timedelta(days=i)
            session_closes = view.closes(session_date=session_date)

            if len(session_closes.columns) != 101:
                continue

            for symbol in view.symbols:
                close_price = float(session_closes[symbol].to_list()[0])
                season_close_prices[symbol].append(close_price)

        seasonal_returns: dict[str, float] = {}
        for symbol, closes in season_close_prices.items():
            if len(closes) < self._window:
                continue
            for i in range(len(closes)):
                current_close = closes[i]
                next_close = closes[(i + 1) % len(closes)]
                return_ = (next_close - current_close) / current_close
                season = season_map.get(i, "Other")
                if season not in seasonal_returns:
                    seasonal_returns[season] = []
                seasonal_returns[season].append(return_)

        top_season = max(seasonal_returns.keys(), key=lambda s: sum(seasonal_returns[s]))
        picks = [symbol for symbol in view.symbols
                 if any(closes[-1] >= max(closes) - 0.05 for closes in season_close_prices.values())]

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