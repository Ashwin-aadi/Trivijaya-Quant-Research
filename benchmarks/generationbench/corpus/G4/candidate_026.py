from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the persistent relationship between market trends and volatility "
        "in the Indian equity market. It aims to capture profits from trending markets while mitigating "
        "risks associated with high volatility by scaling positions based on historical volatility levels."
    )

    def __init__(self, trend_window: int = 50, vol_window: int = 20, top_n: int = 30) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._trend_window + 1).sort("session_date")
        if closes.height < self._vol_window + 2:
            return Signal(information_available_at=stamp, weights={})

        trends: list[tuple[str, float]] = []
        volatilities: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            trend = _trend_direction(closes[symbol].to_list(), self._trend_window)
            volatility = _rolling_volatility(
                closes[symbol].drop_nulls().to_list(), self._vol_window
            )
            trends.append((symbol, trend))
            volatilities[symbol] = volatility

        ranked_trends: list[str] = [t[0] for t in sorted(trends, key=lambda x: abs(x[1]), reverse=True)]
        filtered_symbols = [
            s for s in ranked_trends if volatilities[s] < _rolling_volatility(closes[symbol].to_list(), self._vol_window)
        ][: self._top_n]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_stock = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_stock for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _trend_direction(prices: list[float], window: int) -> float:
    ma = sum(prices[-window:]) / window
    if prices[-1] > ma:
        return 1.0
    elif prices[-1] < ma:
        return -1.0
    else:
        return 0.0


def _rolling_volatility(prices: list[float], window: int) -> float:
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    volatility = (sum([r**2 for r in returns[-window:]]) / window) ** 0.5
    return volatility