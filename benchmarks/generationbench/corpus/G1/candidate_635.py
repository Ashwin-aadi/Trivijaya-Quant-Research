from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining momentum and volatility can provide a balanced approach to identifying "
        "overbought or oversold conditions. Momentum indicates recent price movement, while "
        "volatility suggests market stability. A composite signal is generated based on both."
    )

    def __init__(self, momentum_window: int = 10, volatility_window: int = 20) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._volatility_window))
        if history.is_empty() or history.height < max(self._momentum_window, self._volatility_window):
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            close_history = [float(v) for v in history[symbol].to_list()]
            if len(close_history) < self._momentum_window:
                continue

            # Momentum score based on the last price change
            momentum_score = (close_history[-1] - close_history[0]) / abs(close_history[0])
            momentum_scores[symbol] = momentum_score

            # Volatility score based on the range of prices
            volatility_range = max(close_history) - min(close_history)
            if volatility_range == 0:
                continue
            volatility_score = (close_history[-1] - close_history[0]) / volatility_range
            volatility_scores[symbol] = volatility_score

        combined_scores: dict[str, float] = {}
        for symbol in view.symbols:
            momentum_val = momentum_scores.get(symbol, 0.0)
            volatility_val = volatility_scores.get(symbol, 0.0)
            combined_scores[symbol] = (momentum_val + volatility_val) / 2

        sorted_symbols = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_symbols[:5]]

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