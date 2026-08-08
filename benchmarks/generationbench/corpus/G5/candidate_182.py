from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the idea that assets with higher "
        "recent volatility are more likely to continue their recent trends. By scaling "
        "the trend by its recent volatility, we can take larger positions in more volatile "
        "assets and smaller positions in less volatile ones."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        volatilities: dict[str, float] = {}
        trends: dict[str, tuple[float, float]] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, self._window)]
            volatility = (sum([r**2 for r in returns]) / self._window)**0.5
            volatilities[symbol] = volatility

            if len(prices) < 2:
                continue
            trend = prices[-1] - prices[0]
            trends[symbol] = (trend, volatility)

        sorted_symbols = [s for s in sorted(trends.items(), key=lambda item: -item[1][0])]
        
        top_symbols = sorted_symbols[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        total_volatility = sum(v[1] for v in trends.values())
        if total_volatility == 0:
            return Signal(information_available_at=stamp, weights={})

        weights: dict[str, float] = {}
        for symbol, (trend, volatility) in top_symbols:
            weight = trend / total_volatility
            weights[symbol] = max(0.0, min(weight, 1.0))  # Clamp to [0, 1]

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest