from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of high range compression, market prices move less in absolute terms "
        "but with similar volatility. This often signals a breakout or reversal in the near future."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            high_min = hist.select(pl.col("high").min()).item()
            low_max = hist.select(pl.col("low").max()).item()
            range_compression_score = (high_min - low_max) / max(hist["close"].std().item(), 1e-6)
            range_compression_scores[symbol] = range_compression_score

        top_symbols = sorted(range_compression_scores, key=range_compression_scores.get, reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest