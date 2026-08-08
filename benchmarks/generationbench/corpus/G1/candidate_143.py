from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeTrendVolatility(Strategy):
    rationale = (
        "A combination of upward trend and low volatility suggests that the stock is in a "
        "stable and positive momentum phase. This dual characteristic can indicate a good entry point."
    )

    def __init__(self, trend_window: int = 30, vol_window: int = 20) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._trend_window, self._vol_window))
        if history.is_empty() or history.height < max(self._trend_window, self._vol_window):
            return Signal(information_available_at=stamp, weights={})

        trend_scores: dict[str, float] = {}
        volatilities: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            trend_score = _calculate_trend_score(closes, self._trend_window)
            volatilities[symbol] = _calculate_volatility(closes, self._vol_window)

            if trend_score > 0:
                trend_scores[symbol] = min(1.0, -trend_score / 2)  # Scale to [0, 1]

        filtered_symbols = [
            s for s in trend_scores.keys() if volatilities[s] < 1.0 and trend_scores[s] > 0
        ]
        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_trend_score(prices: list[float], window: int) -> float:
    shifted_prices = [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))]
    mean_return = sum(shifted_prices[-window:]) / min(len(shifted_prices), window)
    return mean_return


def _calculate_volatility(prices: list[float], window: int) -> float:
    shifted_prices = [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))]
    volatility = (sum([p ** 2 for p in shifted_prices[-window:]]) / min(len(shifted_prices), window)) ** 0.5
    return volatility