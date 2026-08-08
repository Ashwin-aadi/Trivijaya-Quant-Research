from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Utilizing historical patterns of stock performance around specific festivals and quarter-ends, "
        "this strategy aims to capitalize on predictable market behavior through calendar-based timing."
    )

    def __init__(self, lookback_years: int = 5, top_n_stocks: int = 30) -> None:
        self._lookback_years = lookback_years
        self._top_n_stocks = top_n_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)
        if history.height < self._lookback_years * 252:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        top_stocks: list[str] = []

        for symbol in symbols:
            if symbol not in history["symbol"].to_list():
                continue

            closes = history.filter(pl.col("symbol") == symbol)["adj_close"]
            returns = (closes / closes.shift(1) - 1).sort(descending=True)
            volume = history.filter(pl.col("symbol") == symbol)["volume"]

            recent_returns = returns[-252:].to_list()
            same_period_last_year = returns[-504:-252].to_list()

            if len(recent_returns) != 252 or len(same_period_last_year) != 252:
                continue

            avg_return_recent = sum(recent_returns)
            avg_return_last_year = sum(same_period_last_year)

            return_diff = avg_return_recent - avg_return_last_year
            trading_volume = volume[-1]

            if return_diff > 0 and trading_volume > 0:
                top_stocks.append(symbol)

        top_stocks = top_stocks[: self._top_n_stocks]
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest