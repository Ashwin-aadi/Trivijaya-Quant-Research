from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy leverages volatility-scaled trend following to capture profitable "
        "trend movements while managing risk during volatile periods. By scaling positions based "
        "on current market volatility, it aims to maximize returns while preserving capital."
    )

    def __init__(self, sma_window: int = 50, atr_window: int = 14, top_n: int = 30) -> None:
        self._sma_window = sma_window
        self._atr_window = atr_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._atr_window)

        if closes.height < self._atr_window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate ATR
        history = view.history()
        high = history["high"]
        low = history["low"]
        close_lag = history["close"].shift(1)
        true_range = (high - low).alias("tr") + (pl.col("close") - high).alias("tr2") + (close_lag - low).alias("tr3")
        a_true_range = true_range.max().alias("atr")

        # Calculate 50-day SMA
        sma = history.select(
            pl.col("symbol").alias("symbol"),
            pl.col("adj_close").mean().over(pl.col("session_date").rank(method="dense", descending=False)).alias("sma")
        ).sort("sma", descending=True).select("symbol")

        # Calculate deviation from SMA
        sma_deviation = (closes - closes.shift(-1)).with_columns(
            pl.col("adj_close") / a_true_range * 100.0
        ).select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close") - sma["sma"]) / a_true_range * 100.0.alias("deviation")
        )

        # Rank symbols by deviation and select top N
        ranked_symbols = sma_deviation.sort("deviation", descending=True).select(
            pl.col("symbol").head(self._top_n)
        ).to_dict(as_series=False)

        if not ranked_symbols["symbol"]:
            return Signal(information_available_at=stamp, weights={})

        # Scale positions based on ATR
        atr = a_true_range[-1]
        max_position_size = 5.0 / self._top_n if atr > 2 * history["close"].std() else 2.0 / self._top_n

        weights = {s: max_position_size for s in ranked_symbols["symbol"]}

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest