from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityEffect(Strategy):
    rationale = (
        "Stocks in India may exhibit seasonality effects due to various factors such as "
        "government policies, festivals, and corporate earnings. By identifying stocks that "
        "tend to outperform during specific months, we can take advantage of these patterns."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = history["symbol"].to_list()
        closes = view.closes().transpose()

        seasonal_returns: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            returns = [(prices[i + 1] - prices[i]) / prices[i] for i in range(len(prices) - 1)]
            avg_return = sum(returns) / len(returns)
            seasonal_returns[symbol] = avg_return

        sorted_symbols = [
            k for k, v in sorted(seasonal_returns.items(), key=lambda item: item[1], reverse=True)
        ]
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