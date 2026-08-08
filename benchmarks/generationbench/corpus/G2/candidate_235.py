from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the high and low prices of a stock are increasingly "
        "concentrated over time. This can indicate that the market is losing its volatility and "
        "could be due for a breakout in either direction. High range compression may suggest an "
        "uptrend, while low range compression could indicate a downtrend."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbol_ranges = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).select(
                pl.col("session_date"), (pl.col("high") - pl.col("low")).alias("range")
            )
            max_range = df.sort("session_date").tail(1)["range"].to_list()[0]
            average_range = df.select((pl.col("range").mean()).alias("avg_range"))[
                "avg_range"
            ].to_list()[0]

            if max_range == 0:
                continue
            range_ratio = average_range / max_range

            symbol_ranges.append(
                {
                    "symbol": symbol,
                    "average_range": average_range,
                    "max_range": max_range,
                    "range_ratio": range_ratio,
                }
            )

        if not symbol_ranges:
            return Signal(information_available_at=stamp, weights={})

        sorted_ranges = sorted(symbol_ranges, key=lambda x: x["range_ratio"], reverse=True)
        top_5_symbols = [r["symbol"] for r in sorted_ranges[:5]]

        weight = 1.0 / len(top_5_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_5_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest