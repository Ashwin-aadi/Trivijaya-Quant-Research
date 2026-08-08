from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of range compression, volatility tends to increase, providing "
        "opportunities for profit. By identifying symbols with reduced price movement, we can "
        "focus on those that may experience higher volatility in the near future."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        compression_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            highs = [float(v) for v in history[symbol + "_high"].drop_nulls().to_list()]
            lows = [float(v) for v in history[symbol + "_low"].drop_nulls().to_list()]
            if len(highs) < self._window or len(lows) < self._window:
                continue
            range_compression = (max(highs) - min(lows)) / max(highs, default=1)
            if range_compression <= self._threshold:
                compression_scores[symbol] = range_compression

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in compression_scores:
                continue
            picks.append(symbol)

        picks = picks[:5]
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