from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price-level reversion suggests that prices revert to a mean level over time. "
        "Using a trailing reference (e.g., 20-day simple moving average) can identify "
        "overbought or oversold conditions, which may indicate reversal opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = [float(c) for c in view.closes().select(symbols).to_dict()["session_date"]]
        ma = sum(closes[-self._window:]) / self._window

        weights: dict[str, float] = {}
        for symbol in symbols:
            close = history.filter(pl.col("symbol") == symbol).select("adj_close").to_series()
            if close.is_empty():
                continue
            if closes[-1][symbols.index(symbol)] < ma:
                weights[symbol] = 0.75
            elif closes[-1][symbols.index(symbol)] > ma:
                weights[symbol] = 0.25

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items() if w > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest