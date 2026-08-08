from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Seasonal effects can be exploited by investing in stocks that historically perform "
        "well during certain times of the year. For instance, tourism and retail sectors may "
        "see higher returns during holiday seasons."
    )

    def __init__(self, season: int = 12, top_n: int = 5) -> None:
        self._season = season
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._season * 365)

        if history.height < self._season * 365:
            return Signal(information_available_at=stamp, weights={})

        seasonal_returns = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_prices = [float(v) for v in history[symbol].to_list()]
            if len(close_prices) < self._season * 365:
                continue

            # Calculate returns and adjust for seasonal lag
            adjusted_close_prices = [close_prices[i] / close_prices[i - 365] for i in range(365, len(close_prices))]
            season_returns = sum([adjusted_close_prices[i] - 1.0 for i in range(self._season)]) / self._season

            seasonal_returns[symbol] = season_returns

        # Select top N symbols with highest returns
        sorted_symbols = sorted(seasonal_returns.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_symbols[:self._top_n]]

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