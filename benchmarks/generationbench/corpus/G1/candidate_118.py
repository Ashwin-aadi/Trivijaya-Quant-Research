from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit strong seasonality, with prices "
        "tending to rise or fall around specific times of the year. By identifying these "
        "patterns, we can generate profitable trades based on historical price movements."
    )

    def __init__(self, window: int = 365, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].unique().to_list()]

        signal_strength: dict[str, float] = {}
        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol)
            closes = df.select(["session_date", "adj_close"]).sort(by="session_date")
            yearly_closes = []
            current_year = None
            for i in range(1, len(closes)):
                if not current_year or closes[i - 1]["session_date"].year != current_year:
                    current_year = closes[i - 1]["session_date"].year
                    yearly_closes.append(closes[i - 1]["adj_close"])
                else:
                    yearly_closes[-1] = (yearly_closes[-1] + closes[i]["adj_close"]) / 2

            last_year_close = yearly_closes[-1]
            if len(yearly_closes) < 3:  # Not enough years to make a meaningful comparison
                continue

            current_year_performance = (closes["adj_close"][-1] - closes["adj_close"][0]) / closes["adj_close"][0]
            last_year_performance = (last_year_close - yearly_closes[0]) / yearly_closes[0]

            if current_year_performance > 2 * last_year_performance:
                signal_strength[symbol] = max(current_year_performance, 1)

        top_symbols = sorted(signal_strength.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        weights = {symbol: strength / self._top_n for symbol, strength in top_symbols}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest