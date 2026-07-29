from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price action is consolidating, which can precede "
        "a breakout or a trend reversal. By identifying symbols with reduced volatility over "
        "the last 20 days, we aim to capture potential breakout opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        compression_scores = []
        for symbol in view.symbols:
            df = history.select(
                pl.col("symbol"), pl.col("session_date"), (pl.col("high") - pl.col("low")).alias("range")
            ).filter(pl.col("symbol") == symbol)
            if df.height < self._window:
                continue
            range_series = [float(v) for v in df["range"].drop_nulls().to_list()]
            avg_range = sum(range_series) / len(range_series)
            recent_ranges = [r for r in range_series[-self._window:]]
            recent_avg_range = sum(recent_ranges) / self._window
            score = (avg_range - recent_avg_range) / avg_range if avg_range > 0 else float('inf')
            compression_scores.append((symbol, score))

        sorted_scores = sorted(compression_scores, key=lambda x: x[1], reverse=True)
        picks = [symb for symb, _ in sorted_scores[:5]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest