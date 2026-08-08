from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating and could be due for a "
        "breakout. This strategy identifies symbols with high dispersion in their recent range, "
        "suggesting potential for significant movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            open_values = [float(v) for v in df["open"].to_list()]
            close_values = [float(v) for v in df["close"].to_list()]
            if len(open_values) < self._window or len(close_values) < self._window:
                continue
            high_value = max([float(v) for v in df["high"].to_list()])
            low_value = min([float(v) for v in df["low"].to_list()])
            range_width = high_value - low_value
            dispersion = sum(abs(o - c) for o, c in zip(open_values, close_values)) / self._window
            if dispersion > range_width * 0.5:
                picks.append(symbol)

        picks = picks[:10]
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