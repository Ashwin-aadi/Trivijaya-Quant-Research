from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Stocks experiencing range compression indicate a period of consolidation. "
        "Such periods often precede breakout moves in either direction, making them valuable for "
        "entry points."
    )

    def __init__(self, window: int = 30, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_vals = [float(v) for v in history[symbol].select("open").to_list()]
            close_vals = [float(v) for v in history[symbol].select("close").to_list()]

            high_low_diffs = [
                (h - l) / max(l, h)
                if h != l
                else 0.0
                for h, l in zip([*open_vals[1:], close_vals[:-1]], open_vals[:-1])
            ]

            if len(high_low_diffs) < self._window:
                continue

            mean_range = sum(high_low_diffs[-self._window:]) / self._window
            score = (mean_range - min(high_low_diffs)) / max(0.0001, mean_range)

            range_compression_scores[symbol] = score

        top_symbols = sorted(range_compression_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest