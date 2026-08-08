from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Stocks often exhibit seasonality effects where performance varies by month. "
        "Identifying and exploiting these patterns can generate alpha."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate monthly mean returns
        month_mean_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or history.is_empty():
                continue

            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            month_dates = history["session_date"].dt.month.to_list()
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            monthly_returns = [returns[month_dates.index(m)] for m in set(month_dates)]

            if len(monthly_returns) > 0:
                month_mean_returns[symbol] = sum(monthly_returns) / len(monthly_returns)

        # Sort by mean return
        sorted_symbols = sorted(month_mean_returns, key=month_mean_returns.get, reverse=True)
        top_symbols = sorted_symbols[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest