from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines a momentum component with a volatility filter. "
        "Momentum signals strong trend continuation, while low volatility suggests "
        "calm market conditions more conducive to holding positions."
    )

    def __init__(self, momentum_window: int = 10, volatility_window: int = 5) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)

        if history.is_empty() or history.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes or len(closes) < self._momentum_window + self._volatility_window:
                continue

            close_values = [float(v) for v in history[symbol].to_list()]
            momentum_score = _calculate_momentum(close_values, self._momentum_window)
            volatility_score = _calculate_volatility(close_values, self._volatility_window)

            if momentum_score > 0 and volatility_score < 1:
                momentum_scores[symbol] = momentum_score
                volatility_scores[symbol] = volatility_score

        selected_symbols = [s for s in momentum_scores.keys() if volatility_scores[s] >= 0.5]
        weights = {s: (momentum_scores[s] + volatility_scores[s]) / len(selected_symbols) for s in selected_symbols}

        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol, weight in weights.items()}
        )


def _calculate_momentum(close_values: list[float], window: int) -> float:
    recent_prices = close_values[-window:]
    mean_price = sum(recent_prices) / len(recent_prices)
    momentum_score = (close_values[-1] - mean_price) / mean_price
    return max(0, min(momentum_score, 1))


def _calculate_volatility(close_values: list[float], window: int) -> float:
    recent_prices = close_values[-window:]
    std_deviation = pl.Series(recent_prices).std()
    volatility_score = (std_deviation - min(recent_prices)) / max(recent_prices)
    return max(0, min(volatility_score, 1))


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest