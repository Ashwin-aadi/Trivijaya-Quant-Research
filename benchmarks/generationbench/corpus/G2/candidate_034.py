from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Stock markets often exhibit predictable seasonal patterns. For instance, some stocks may "
        "perform better at the beginning of the year due to end-of-year profit-taking or after the "
        "annual earnings reports in January. Identifying and trading these seasonal effects can lead "
        "to consistent returns."
    )

    def __init__(self, lookback_years: int = 5) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)  # Approximate trading days in a year
        if history.is_empty() or history.height < self._lookback_years * 252:
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(["symbol", "session_date", "adj_close"])
        seasonal_effects: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.column_names:
                continue
            symbol_data = closes.filter(pl.col("symbol") == symbol)
            if symbol_data.height < self._lookback_years * 252:
                continue

            yearly_returns: list[float] = []
            for i in range(self._lookback_years):
                start_date = stamp - date(1, 1, 1) + date(i + 1, 1, 1)
                end_date = start_date + date(1, 1, 1) - date(0, 1, 1)

                yearly_data = symbol_data.filter(
                    (pl.col("session_date") >= start_date) & (pl.col("session_date") < end_date)
                )
                if not yearly_data.height:
                    continue

                open_price = float(yearly_data["adj_close"][0])
                close_price = float(yearly_data["adj_close"][-1])
                yearly_returns.append((close_price - open_price) / open_price)

            avg_return = sum(yearly_returns) / len(yearly_returns)
            seasonal_effects[symbol] = avg_return

        sorted_symbols = [symbol for symbol, _ in sorted(seasonal_effects.items(), key=lambda item: item[1], reverse=True)]
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in sorted_symbols[:5]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest