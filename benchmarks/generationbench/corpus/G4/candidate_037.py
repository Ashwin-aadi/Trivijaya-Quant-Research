from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion involves identifying stocks that have recently "
        "experienced significant price deviations from their moving averages. These "
        "deviations are exploited by buying underpriced stocks and selling overpriced ones."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = list(closes.columns)
        sma_series = [closes[symbol].mean().item() for symbol in symbols]
        deviations = [(float(closes[symbol][-1]) - sma).item() for symbol, sma in zip(symbols, sma_series)]

        underpriced_symbols = sorted(zip(symbols, deviations), key=lambda x: abs(x[1]))[: self._top_n]
        overpriced_symbols = sorted(zip(symbols, deviations), key=lambda x: abs(x[1]), reverse=True)[: self._top_n]

        underpriced_weights = {s: 0.6 for s, _ in underpriced_symbols}
        overpriced_weights = {s: 0.3 for s, _ in overpriced_symbols}

        weights = {**underpriced_weights, **overpriced_weights}
        return Signal(information_available_at=stamp, weights={k: v for k, v in weights.items() if v > 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest