from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a market is consolidating, which can be a precursor to "
        "a breakout or reversal. By identifying symbols where the range has compressed recently, "
        "we aim to capture potential breakout opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            open_vals = [float(v) for v in history.filter(pl.col("symbol") == symbol)["open"].drop_nulls().to_list()]
            close_vals = [float(v) for v in history.filter(pl.col("symbol") == symbol)["close"].drop_nulls().to_list()]

            if len(open_vals) < self._window or len(close_vals) < self._window:
                continue

            high = max([float(v) for v in history.filter(pl.col("symbol") == symbol)["high"].drop_nulls().to_list()])
            low = min([float(v) for v in history.filter(pl.col("symbol") == symbol)["low"].drop_nulls().to_list()])

            recent_open_range = (max(open_vals[-self._window:]) - min(open_vals[-self._window:])) / high
            recent_close_range = (max(close_vals[-self._window:]) - min(close_vals[-self._window:])) / low

            range_compression_scores[symbol] = abs(high - low) / (high + low)

        sorted_scores = sorted(range_compression_scores.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, score in sorted_scores[:5]]

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