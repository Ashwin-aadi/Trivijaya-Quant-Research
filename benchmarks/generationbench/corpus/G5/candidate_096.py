from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy selects the top-performing assets based on their "
        "10-day momentum to capture short-term trending behavior in the market."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        top_n = min(self._window, len(view.symbols))
        momentum_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns or history[symbol].is_empty():
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window + 1:
                continue
            momentum = (prices[-1] / prices[0]) - 1.0
            momentum_scores[symbol] = momentum

        top_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weights = {symbol: 1.0 / len(top_symbols) for symbol, _ in top_symbols}
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest