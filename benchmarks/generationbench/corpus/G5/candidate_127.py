from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a period of low volatility and may suggest that "
        "the market is consolidating before a potential breakout. Investing in symbols with "
        "compressed ranges can help capture any subsequent price movements."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol)
            highs = [float(v) for v in hist["high"].to_list()]
            lows = [float(v) for v in hist["low"].to_list()]
            if len(highs) < self._window or len(lows) < self._window:
                continue
            high_min = min(highs)
            low_max = max(lows)
            range_compression_score = (high_min - low_max) / ((max(highs) - min(lows)) + 1e-8)
            range_compression_scores[symbol] = range_compression_score

        sorted_symbols = [
            s for _, s in sorted(range_compression_scores.items(), key=lambda item: item[1], reverse=True)
        ][: self._top_n]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).collect().row(0)[0]
    assert isinstance(newest, date)
    return newest