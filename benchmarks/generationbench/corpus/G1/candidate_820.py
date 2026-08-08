from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency for asset prices to revert to "
        "their mean levels over time. By identifying symbols that have deviated significantly "
        "from their average price level, we can generate buy or sell signals."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 1.5) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window).sort("session_date").tail(self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        symbols = [symbol for symbol in view.symbols if symbol in closes]
        
        mean_adj_close = sum(closes) / len(closes)
        std_dev_adj_close = ((sum((x - mean_adj_close) ** 2 for x in closes)) / len(closes)) ** 0.5

        z_scores = [(c - mean_adj_close) / std_dev_adj_close if std_dev_adj_close > 0 else 0 for c in closes]

        candidates: list[str] = []
        for symbol, z_score in zip(symbols, z_scores):
            if abs(z_score) >= self._z_score_threshold:
                candidates.append(symbol)

        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in candidates}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest