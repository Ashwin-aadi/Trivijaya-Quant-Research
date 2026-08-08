from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting is a strategy where we overweight stocks with lower historical volatility. "
        "The rationale behind this approach is that firms with lower beta tend to outperform over time."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        closes = history.select(["session_date"] + symbols).sort("session_date")
        returns = (closes[symbols] / closes[symbols].shift(1) - 1.0).rename({col: f"ret_{col}" for col in symbols})
        volatilities = [float(v.mean()) for v in returns.to_dicts()]
        weighted_symbols = {s: volatilities[i] for i, s in enumerate(symbols)}
        sorted_weights = sorted(weighted_symbols.items(), key=lambda x: x[1])

        top_5_symbols = [s for s, _ in sorted_weights[:5]]
        weight = 1.0 / len(top_5_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_5_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest