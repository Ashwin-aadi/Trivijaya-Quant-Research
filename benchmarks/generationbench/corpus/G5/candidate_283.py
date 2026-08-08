from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price action is becoming less volatile and "
        "concentrated within a narrower range. This suggests potential breakout opportunities "
        "in either direction, making it an opportune time to enter positions."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression = (history["high"] - history["low"]) / history["close"].shift(-1).alias("range_ratio")
        history = history.with_columns(range_compression)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        ranges = [float(v) for v in history.select(pl.col("range_ratio")).to_list()[0]]
        average_range = sum(ranges) / len(ranges)

        low_range_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_history = history.filter(pl.col("symbol") == symbol)
            symbol_ranges = [float(v) for v in symbol_history.select(pl.col("range_ratio")).to_list()[0]]
            if len(symbol_ranges) < self._window:
                continue
            if min(symbol_ranges) <= average_range * 1.5:  # Assuming a threshold of 1.5 times the average range
                low_range_symbols.append(symbol)

        if not low_range_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(low_range_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_range_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest