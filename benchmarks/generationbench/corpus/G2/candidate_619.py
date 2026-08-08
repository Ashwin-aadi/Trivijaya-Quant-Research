from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's high and low prices are more compressed over "
        "time. This suggests that the stock is consolidating or experiencing reduced volatility, "
        "and could be due to institutional buying or selling pressures. We can identify such stocks "
        "by observing changes in the range (high - low) over consecutive periods."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        ranges = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("adj_close").shift(-1) - pl.col("adj_open")).abs().alias("change")
            )
            .group_by("symbol")
            .agg([pl.mean("range").alias("mean_range"), pl.max("change").alias("max_change")])
        )

        # Identify symbols with significant range compression
        compressed_symbols = ranges.filter(
            (ranges["mean_range"] / ranges["max_change"]) > self._threshold
        ).select("symbol").to_series().to_list()

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest