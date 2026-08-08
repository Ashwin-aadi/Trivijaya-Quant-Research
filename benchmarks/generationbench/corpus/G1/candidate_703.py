from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression is a period during which the price movement is more confined within "
        "a narrow range, suggesting that market sentiment might be shifting. By identifying such "
        "periods, we can anticipate potential breakout or trend continuation."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = history.select(["symbol", "session_date", pl.col("adj_close").alias("close")])

        # Calculate daily range for each symbol
        ranges = (
            closes
            .group_by("symbol")
            .agg(
                (pl.col("close").max() - pl.col("close").min()).alias("range"),
            )
        )

        # Find symbols with the highest range compression
        top_compressed: list[str] = []
        for symbol in symbols:
            if symbol not in ranges.columns or float(ranges.get_symbol(symbol)["range"]) < 0.1 * self._window:
                continue
            top_compressed.append(symbol)

        if not top_compressed:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_compressed)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_compressed}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest