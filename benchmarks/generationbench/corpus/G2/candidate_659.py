from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy combines short-term and long-term momentum to identify "
        "overbought or oversold conditions in the market. Stocks that are both "
        "outperforming their peers over a short term and have been trending positively "
        "in the longer term are expected to continue performing well."
    )

    def __init__(self, short_window: int = 10, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._long_window + self._short_window - 1)

        if history.height < self._long_window + self._short_window - 1:
            return Signal(information_available_at=stamp, weights={})

        short_momentum: dict[str, float] = {}
        long_momentum: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]

            # Short-term momentum
            short_returns = [
                (closes[i] - closes[i - self._short_window]) / closes[i - self._short_window]
                for i in range(self._short_window, len(closes))
            ]
            if all([r > 0.01 for r in short_returns]):
                short_momentum[symbol] = max(short_returns)

            # Long-term momentum
            long_returns = [
                (closes[i] - closes[i - self._long_window]) / closes[i - self._long_window]
                for i in range(self._long_window, len(closes))
            ]
            if all([r > 0.02 for r in long_returns]):
                long_momentum[symbol] = max(long_returns)

        # Find symbols that meet both short-term and long-term criteria
        combined_scores = {s: short_momentum[s] * long_momentum.get(s, 0) for s in short_momentum}
        sorted_symbols = sorted(combined_scores.items(), key=lambda x: -x[1])

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbol, score = sorted_symbols[0]
        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest