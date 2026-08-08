from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression20d(Strategy):
    rationale = (
        "During periods of high uncertainty or news-driven events, stock prices may spread out "
        "(increase in dispersion), while during calm periods, they tend to trade within a narrower range. "
        "This strategy exploits this pattern by entering long positions when the price breaks above the upper boundary "
        "of its recent range and short positions when it falls below the lower boundary."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_range = (
            history.select(
                pl.col("high").subtract(pl.col("low")).alias("range")
            ).group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
            .select("avg_range")
        )
        latest_closes = view.latest_close()
        signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in avg_range.height or symbol not in latest_closes.keys():
                continue

            avg_range_val = avg_range.select(pl.col("avg_range").filter(pl.col("symbol") == symbol)).item()
            high = history.select(pl.col("high").filter(pl.col("symbol") == symbol)).item()
            low = history.select(pl.col("low").filter(pl.col("symbol") == symbol)).item()
            close = latest_closes[symbol]

            upper_boundary = avg_range_val * 1.5 + close
            lower_boundary = close - (avg_range_val * 1.5)

            if high > upper_boundary:
                signals[symbol] = 0.05

            elif low < lower_boundary:
                signals[symbol] = -0.05

        return Signal(information_available_at=stamp, weights={s: w for s, w in signals.items() if w != 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest