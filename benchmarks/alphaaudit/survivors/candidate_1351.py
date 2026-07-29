from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased market volatility and may signal "
        "an upcoming breakout or significant price movement. By identifying symbols with "
        "compressed ranges, we can position our portfolio to benefit from potential "
        "price movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).mean().alias("range"),
                (pl.col("close").std() / 2.0).alias("volatility"),
            )
            .with_columns(
                (pl.col("range") / pl.col("volatility")).alias("ratio")
            )
        )

        if range_compression.is_empty():
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols = (
            range_compression.sort("ratio", descending=True)
            .head(5)["symbol"]
            .to_list()
        )

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