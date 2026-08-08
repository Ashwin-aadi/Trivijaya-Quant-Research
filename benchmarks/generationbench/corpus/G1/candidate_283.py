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

        range_compression = (
            (history["high"] - history["low"]) / history["close"].shift(-1)
        ).alias("range_ratio")
        range_ratio_series = history.with_columns(range_compression).select(
            pl.col("range_ratio").to_list()
        )
        if len(range_ratio_series.to_list()[0]) < self._window:
            return Signal(information_available_at=stamp, weights={})

        ranges = [float(v) for v in range_ratio_series.to_list()[0]]
        average_range = sum(ranges) / len(ranges)

        symbols_with_lowest_range = [
            symbol
            for symbol in view.symbols
            if min(
                (float(history[history["symbol"] == symbol]["range_ratio"].to_list()[-1])
                 for history in [view.history(lookback=self._window)]),
                default=0.0,
            )
        ]

        if not symbols_with_lowest_range:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_lowest_range)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols_with_lowest_range}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest