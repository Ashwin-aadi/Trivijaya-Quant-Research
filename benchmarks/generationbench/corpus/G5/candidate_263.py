from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term momentum with long-term trend following can capture both "
        "rapid price movements and persistent trends, potentially leading to more robust returns."
    )

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_window)
        if closes.height < self._long_window or not all(symbol in closes.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        short_returns: dict[str, float] = {}
        long_returns: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._long_window:
                continue

            short_return = (values[-1] / values[-self._short_window - 1] - 1.0)
            long_return = (values[-1] / values[0] - 1.0)

            short_returns[symbol] = short_return
            long_returns[symbol] = long_return

        # Filter symbols with both returns above a certain threshold
        combined_scores = {
            symbol: short_returns[symbol] + long_returns[symbol]
            for symbol in short_returns.keys() & long_returns.keys()
            if short_returns[symbol] > 0.05 and long_returns[symbol] > 0.02
        }

        if not combined_scores:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in [t[0] for t in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest