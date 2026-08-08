from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumAndVolatility(Strategy):
    rationale = (
        "This strategy seeks to identify stocks with both high recent momentum and low "
        "volatility. High momentum indicates strong price movement, while low volatility suggests "
        "the stock is less risky and more predictable."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 30) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)

        if history.is_empty() or history.height < (self._momentum_window + self._volatility_window):
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns or "symbol" not in history.columns:
                continue

            # Calculate momentum
            close_prices = [float(v) for v in history[symbol].to_list()]
            last_price = close_prices[-1]
            max_close = max(close_prices)
            min_close = min(close_prices)

            if last_price >= (max_close * 0.95):
                momentum_scores[symbol] = max_close - min_close

            # Calculate volatility
            log_returns = [((close / prev_close) - 1) for close, prev_close in zip(close_prices[1:], close_prices[:-1])]
            std_deviation = pl.DataFrame({"return": log_returns})["return"].std()
            if not std_deviation.is_null():
                volatility_scores[symbol] = std_deviation

        # Filter out symbols with no momentum or volatility
        momentum_filtered = {k: v for k, v in momentum_scores.items() if v > 0}
        volatility_filtered = {k: v for k, v in volatility_scores.items() if not v.is_null()}
        
        common_symbols = set(momentum_filtered.keys()) & set(volatility_filtered.keys())
        
        # If no common symbols are found
        if not common_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to all selected stocks
        weight = 1.0 / len(common_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in common_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest