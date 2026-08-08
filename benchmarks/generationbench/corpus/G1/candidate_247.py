from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates reduced volatility and increased consolidation. "
        "Stocks that have been consolidating might be due for a breakout or reversal. "
        "By identifying such stocks, we can position our portfolio to benefit from potential reversals."
    )

    def __init__(self, window: int = 30, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.groupby("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("adj_close").last() - pl.col("adj_close").first()).abs().alias("close_change"),
            )
            .with_columns(
                ((pl.col("close_change") / pl.col("range")) < self._threshold).alias("is_compressed")
            )
        )

        compressed_symbols = range_compression.filter(pl.col("is_compressed")).select(["symbol"]).to_dict(False)

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols["symbol"])
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for symbol in compressed_symbols["symbol"]
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().to_list()[0]
    assert isinstance(newest, date)
    return newest