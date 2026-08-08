from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Large price movements (volatility) are often followed by mean-reverting behavior. "
        "By scaling the trend following signal with volatility, we aim to take larger positions in "
        "high-volatility environments where trends persist longer and smaller positions when "
        "volatility is low."
    )

    def __init__(self, window: int = 20, factor: float = 1.5) -> None:
        self._window = window
        self._factor = factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol].to_list()]
            # Calculate daily returns
            returns = [(close_series[i] - close_series[i-1]) / close_series[i-1]
                       for i in range(1, len(close_series))]
            # Calculate the volatility over the window period
            volatilities[symbol] = (sum(abs(r) for r in returns) / self._window) ** 0.5

        if not volatilities:
            return Signal(information_available_at=stamp, weights={})

        mean_return = sum(returns) / len(returns)
        weighted_returns = {s: v * (mean_return + v * self._factor) for s, v in volatilities.items()}

        total_weight = 0
        for symbol, value in sorted(weighted_returns.items(), key=lambda x: -x[1]):
            if total_weight + value > 1.0:
                break
            total_weight += value

        picks = [s for s, _ in weighted_returns.items() if _ > 0]
        weight = (1.0 - total_weight) / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp,
            weights={p: weight + w for p, w in ((s, value) for s, value in weighted_returns.items() if s in picks)}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest