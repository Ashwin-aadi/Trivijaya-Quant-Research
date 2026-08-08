from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating and may soon breakout. "
        "By identifying symbols with reduced price ranges, we can anticipate potential trending behavior."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores: dict[str, float] = {}
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).select(
                pl.col("high"), pl.col("low")
            )
            if df.is_empty():
                continue
            high_values = [float(v) for v in df["high"].to_list()]
            low_values = [float(v) for v in df["low"].to_list()]
            max_high = max(high_values)
            min_low = min(low_values)
            range_compression_score = (max_high - min_low) / sum(
                abs(h - l) for h, l in zip(high_values[:-1], low_values[:-1])
            )
            if not math.isnan(range_compression_score):
                range_compression_scores[symbol] = range_compression_score

        sorted_symbols = [
            symbol
            for _, symbol in sorted(
                range_compression_scores.items(), key=lambda item: item[1]
            )
        ]
        top_n_symbols = sorted_symbols[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest