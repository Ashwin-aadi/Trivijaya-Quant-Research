from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy aims to exploit a combination of momentum and volatility. "
        "Momentum suggests that past winners will continue to outperform, while low volatility can indicate stable underlying fundamentals. "
        "By investing in symbols with strong historical performance and low current volatility, the strategy seeks to capture both trends."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 5) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)

        if history.is_empty() or history.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)
            closes = [float(v) for v in symbol_history["adj_close"].to_list()]
            if len(closes) < self._momentum_window:
                continue
            # Calculate simple moving average (SMA) and subtract it from the most recent close
            sma_last = sum(closes[-self._momentum_window:]) / self._momentum_window - closes[-1]
            momentum_scores[symbol] = sma_last

        volatilities = {}
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)
            if len(symbol_history) < self._volatility_window + 1:
                continue
            # Calculate the log returns and then their standard deviation
            log_returns = [float(v) / v.shift(1) - 1.0 for v in symbol_history["adj_close"].to_list()[-self._volatility_window:]]
            volatility = (sum(log_returns)**2 / self._volatility_window)**0.5
            volatilities[symbol] = volatility

        # Select the top performing symbols with low volatility
        sorted_symbols = [
            s for _, s in sorted(
                momentum_scores.items(), key=lambda x: x[1], reverse=True
            ) if volatilities.get(s, float('inf')) < 0.2
        ][:5]

        weight = 1.0 / len(sorted_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in sorted_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest