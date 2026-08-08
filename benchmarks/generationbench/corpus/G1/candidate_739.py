from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating and may be due for a breakout. "
        "Identifying symbols with reduced price volatility can indicate potential upcoming momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        compressed_ranges = []
        for symbol in symbols:
            data = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close")
            )
            prices = [float(v) for v in data["close"].to_list()]
            high_low_diff = max(prices) - min(prices)
            avg_range = sum(prices[i + 1] - prices[i] for i in range(len(prices) - 1)) / (
                len(prices) - 1
            )
            compressed_ranges.append((symbol, high_low_diff, avg_range))

        # Sort by the highest reduction in price range and average daily range
        sorted_symbols = sorted(
            compressed_ranges,
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )

        top_symbols = [s for s, _, _ in sorted_symbols[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest