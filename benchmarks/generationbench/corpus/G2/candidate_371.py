from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy aims to capture opportunities by combining two unrelated characteristics: "
        "momentum and volatility. Momentum suggests that past winners may continue to outperform, "
        "while low volatility indicates reduced risk in the current market environment."
    )

    def __init__(self, momentum_window: int = 10, vol_window: int = 5) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window + self._volatility_window - 1)
        if closes.height < self._momentum_window + self._volatility_window - 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatilities: dict[str, float] = {}

        for symbol in view.symbols:
            close_prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_prices) < self._momentum_window + self._volatility_window - 1:
                continue

            momentum_score = (
                sum(close_prices[-self._momentum_window:] > close_prices[:-self._momentum_window])
                / self._momentum_window
            )
            vol_score = (pl.col(symbol).std().to_list()[-self._volatility_window:])[-1]
            momentum_scores[symbol] = momentum_score
            volatilities[symbol] = vol_score

        # Filter out symbols with very low volatility to avoid high risk
        threshold = 0.2
        filtered_symbols = {s: (m, v) for s, (m, v) in momentum_scores.items() if volatilities[s] < threshold}

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted(filtered_symbols.items(), key=lambda x: x[1], reverse=True)[:5]
        weights = {s: 0.2 for s, (m, v) in top_symbols}
        weights["cash"] = 0.8 - sum(weights.values())

        return Signal(information_available_at=stamp, weights=dict(weights))


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest