from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumAndVolatility(Strategy):
    rationale = (
        "Combining momentum and low volatility strategies can lead to higher risk-adjusted returns. "
        "Momentum investors buy stocks that have recently performed well, while low volatility strategy "
        "investors prefer stocks with lower price movements. The combination can capture both trends and stabilize portfolio risk."
    )

    def __init__(self, momentum_window: int = 20, vol_window: int = 20) -> None:
        self._momentum_window = momentum_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._vol_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        volatilities: dict[str, float] = {}
        momentum_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._vol_window + 1:
                continue

            # Calculate momentum score as the percentage change over the window period
            last_close = float(values[-1])
            first_close = float(values[0])
            momentum_score = (last_close - first_close) / first_close
            momentum_scores[symbol] = momentum_score

            # Calculate volatility using standard deviation of returns
            returns = [(values[i+1] - values[i]) / values[i] for i in range(len(values)-1)]
            volatilities[symbol] = pl.Series(returns).std()

        sorted_momentum = sorted(momentum_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_volatility = sorted(volatilities.items(), key=lambda x: x[1])

        selected_symbols = [symbol for symbol, _ in sorted_momentum[:5]] + \
                           [symbol for symbol, _ in sorted_volatility][:3]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest