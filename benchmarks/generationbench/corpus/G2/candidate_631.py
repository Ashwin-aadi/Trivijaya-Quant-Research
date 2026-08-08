from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression occurs when a stock's price volatility decreases over a period. "
        "This can indicate that the stock is consolidating and may be due for a breakout in the future. "
        "Identifying such stocks can provide trading opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_series = [float(v) for v in history[symbol + "_high"].to_list()]
            low_series = [float(v) for v in history[symbol + "_low"].to_list()]
            if len(high_series) < self._window or len(low_series) < self._window:
                continue
            high_low_ratio = (max(high_series) - min(low_series)) / max(high_series)
            range_compression_scores[symbol] = high_low_ratio

        # Filter out symbols with the lowest range compression scores
        sorted_symbols = sorted(range_compression_scores.items(), key=lambda x: x[1])
        selected_symbols = [symbol for symbol, score in sorted_symbols[:5]]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest