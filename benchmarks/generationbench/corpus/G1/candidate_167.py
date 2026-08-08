from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeFeature(Strategy):
    rationale = (
        "This strategy identifies stocks that exhibit both a strong recent momentum and a low volatility "
        "level. The combination of these two characteristics aims to capture stocks with high potential for appreciation."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 30) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._volatility_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        volatility_scores = {}

        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue

            close_prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(close_prices) < self._momentum_window:
                continue
            recent_closes = close_prices[-self._momentum_window:]
            momentum_score = (recent_closes[-1] - min(recent_closes)) / max(0.001, max(recent_closes) - min(recent_closes))

            volatility_scores[symbol] = _calculate_volatility(close_prices[: self._volatility_window])

            if symbol in momentum_scores and momentum_score > momentum_scores[symbol]:
                momentum_scores[symbol] = momentum_score

        top_symbols = sorted(momentum_scores.items(), key=lambda x: (x[1], -volatility_scores[x[0]]), reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(prices: list[float]) -> float:
    returns = [(prices[i] - prices[i-1]) / max(0.001, prices[i-1]) for i in range(1, len(prices))]
    volatility = (sum([abs(r) for r in returns]) / len(returns)) ** 0.5
    return volatility