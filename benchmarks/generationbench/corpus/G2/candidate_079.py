from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that when a stock's price moves less than usual over "
        "a period, it could lead to an eventual breakout. This is based on the idea that "
        "tight ranges can build up buying or selling pressure, which may then be released in "
        "the form of a strong move."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        compression_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_low_diffs = (history[symbol]["high"] - history[symbol]["low"]).to_list()
            mean_range = sum(high_low_diffs) / len(high_low_diffs)
            current_range = float(history[symbol][-1]["high"]) - float(
                history[symbol][-1]["low"]
            )
            compression_ratio = current_range / mean_range
            if compression_ratio < 1.0:
                compression_scores[symbol] = (mean_range, current_range)

        # Filter out symbols that do not meet the threshold for range compression
        filtered_symbols = [
            s for s in compression_scores.keys() if compression_scores[s][1] / compression_scores[s][0] < self._threshold
        ]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected symbol
        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest