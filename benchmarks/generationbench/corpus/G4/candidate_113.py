from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "The strategy aims to capitalize on breakout continuation patterns by identifying "
        "breakouts from support or resistance levels and then trading in the direction of the "
        " breakout. The inclusion of volume analysis and a trend indicator helps filter out false "
        " breakouts, enhancing the reliability of trades."
    )

    def __init__(self, lookback_days: int = 50, top_n: int = 5) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.height < self._lookback_days:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily highs and lows
        high_values = (
            history.group_by("symbol")
                   .agg((pl.col("high").max().alias("highest_high")))
                   .join(history.select(["symbol", "session_date"]), on="symbol")
        )
        low_values = (
            history.group_by("symbol")
                    .agg((pl.col("low").min().alias("lowest_low")))
                    .join(history.select(["symbol", "session_date"]), on="symbol")
        )

        # Identify breakouts
        breakout_highs = high_values.filter(
            (pl.col("high") > pl.col("highest_high")) &
            (pl.col("session_date") == stamp)
        )
        breakout_lows = low_values.filter(
            (pl.col("low") < pl.col("lowest_low")) &
            (pl.col("session_date") == stamp)
        )

        if not breakout_highs.height or not breakout_lows.height:
            return Signal(information_available_at=stamp, weights={})

        # Rank candidates based on volume and MACD
        high_candidates = (
            breakout_highs
                   .join(view.closes(lookback=self._lookback_days), on="symbol")
                   .filter(pl.col("session_date") == stamp)
                   .with_column((pl.col("volume") / pl.col("volume").sum()).alias("volume_share"))
                   .sort("volume_share", descending=True)
        )
        low_candidates = (
            breakout_lows
                   .join(view.closes(lookback=self._lookback_days), on="symbol")
                   .filter(pl.col("session_date") == stamp)
                   .with_column((pl.col("volume") / pl.col("volume").sum()).alias("volume_share"))
                   .sort("volume_share", descending=True)
        )

        high_symbols = [row["symbol"] for row in high_candidates.head(self._top_n).rows()]
        low_symbols = [row["symbol"] for row in low_candidates.head(self._top_n).rows()]

        if not high_symbols and not low_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Apply MACD filter
        macd_high = (
            view.history(lookback=self._lookback_days)
                   .filter(pl.col("symbol").is_in(high_symbols))
                   .select(["symbol", "session_date", "close"])
                   .with_column(
                       (pl.col("close") - pl.col("close").shift(self._lookback_days)).alias("macd")
                   )
        )

        macd_low = (
            view.history(lookback=self._lookback_days)
                    .filter(pl.col("symbol").is_in(low_symbols))
                    .select(["symbol", "session_date", "close"])
                    .with_column(
                        (pl.col("close") - pl.col("close").shift(self._lookback_days)).alias("macd")
                    )
        )

        high_symbols = [row["symbol"] for row in macd_high.filter(pl.col("macd") > 0).rows()]
        low_symbols = [row["symbol"] for row in macd_low.filter(pl.col("macd") < 0).rows()]

        if not high_symbols and not low_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Generate signal
        weight = 1.0 / len(high_symbols + low_symbols)
        high_weights = {s: weight for s in high_symbols}
        low_weights = {s: -weight for s in low_symbols}

        final_weights = {**high_weights, **low_weights}
        return Signal(information_available_at=stamp, weights=final_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest