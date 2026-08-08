from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression is a phenomenon where the trading range of a stock narrows "
        "over time. It often indicates an upcoming breakout or consolidation in price action."
    )

    def __init__(self, window: int = 20, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history.select(["symbol", "session_date", "adj_close"])
        recent_highs = history.select(["symbol", "session_date", "high"])
        recent_lows = history.select(["symbol", "session_date", "low"])

        ranges = (
            (recent_highs - recent_lows)
            .group_by("symbol")
            .agg(pl.col("adj_close").max().alias("max_adj_close"),
                 pl.col("adj_close").min().alias("min_adj_close"))
            .with_columns(
                ((pl.col("max_adj_close") - pl.col("min_adj_close")) / (pl.col("max_adj_close") - pl.col("min_adj_close")).shift(1) - 1.0).alias("range_ratio")
            )
        )

        compressed_symbols = ranges.filter(pl.col("range_ratio") < self._threshold).select("symbol").to_series().to_list()

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest