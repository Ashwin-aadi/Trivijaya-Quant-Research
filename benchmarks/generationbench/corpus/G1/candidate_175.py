from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the price action is consolidating in a tight range, "
        "indicating potential for breakout. By identifying stocks with reduced daily price "
        "range over a lookback period, we can find candidates for future moves."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compressed: list[str] = []
        for symbol in view.symbols:
            opens = history.select(pl.col(symbol).alias("open")).to_pandas()["open"]
            closes = history.select(pl.col(symbol).alias("close")).to_pandas()["close"]
            highs = history.select(pl.col(symbol).alias("high")).to_pandas()["high"]
            lows = history.select(pl.col(symbol).alias("low")).to_pandas()["low"]

            daily_ranges = (highs - lows).dropna().values
            mean_range = daily_ranges.mean()

            if (opens[-1] - closes[-1]) / mean_range < self._threshold:
                range_compressed.append(symbol)

        range_compressed = range_compressed[:5]
        if not range_compressed:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(range_compressed)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in range_compressed}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_pandas()["session_date"].iloc[0]
    assert isinstance(newest, date)
    return newest