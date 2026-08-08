from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the difference between the high and low prices "
        "of a stock over a short period is significantly reduced. This often indicates "
        "that the stock is consolidating or preparing for a breakout in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        compression_scores = []
        for symbol in symbols:
            high_low_diffs = (history[symbol]["high"] - history[symbol]["low"]).to_list()
            mean_high_low_diff = sum(high_low_diffs) / len(high_low_diffs)
            current_range = (
                float(history[symbol].select(pl.col("close").max().alias("high")).row(0)[1])
                - float(history[symbol].select(pl.col("close").min().alias("low")).row(0)[1])
            )
            compression_score = mean_high_low_diff / (current_range + 1e-9)
            compression_scores.append((symbol, compression_score))

        sorted_scores = sorted(compression_scores, key=lambda x: x[1], reverse=True)[:5]
        top_symbols = [score[0] for score in sorted_scores]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest