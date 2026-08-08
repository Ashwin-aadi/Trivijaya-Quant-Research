from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion strategy exploits the tendency for stock prices to revert "
        "to their historical means after significant deviations. This is driven by investor "
        "overreactions and temporary mispricings which can be arbitraged."
    )

    def __init__(self, window: int = 20, std_dev_multiplier: float = 2.0) -> None:
        self._window = window
        self._std_dev_multiplier = std_dev_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Compute moving average and Bollinger Bands
        ma = history.select(
            pl.col("adj_close").rolling_mean(self._window).alias(f"ma_{self._window}")
        )
        std_dev = history.select(
            pl.col("adj_close").rolling_std(self._window).alias(f"std_dev_{self._window}")
        )

        # Calculate upper and lower Bollinger Bands
        bands = ma.join(std_dev, on="session_date")
        bands = bands.with_columns(
            (bands[f"ma_{self._window}"] + self._std_dev_multiplier * bands[f"std_dev_{self.window}"]).alias("upper_band"),
            (bands[f"ma_{self.window}"] - self._std_dev_multiplier * bands[f"std_dev_{self.window}"]).alias("lower_band")
        )

        # Identify candidates for long and short positions
        long_candidates = bands.filter(
            pl.col("adj_close") < bands["lower_band"]
        ).select([pl.col("symbol").alias("symbol"), "session_date"])
        
        short_candidates = bands.filter(
            pl.col("adj_close") > bands["upper_band"]
        ).select([pl.col("symbol").alias("symbol"), "session_date"])

        if not long_candidates.height and not short_candidates.height:
            return Signal(information_available_at=stamp, weights={})

        # Rank candidates
        long_ranks = long_candidates.sort(pl.col("adj_close", descending=False)).height
        short_ranks = short_candidates.sort(pl.col("adj_close", descending=True)).height

        weight_long = 1.0 / min(long_ranks, 20)
        weight_short = -1.0 / min(short_ranks, 20)

        weights = {s: weight for s in long_candidates["symbol"].to_list()}
        if short_candidates.height:
            for s in short_candidates["symbol"].to_list():
                weights[s] += weight_short

        return Signal(information_available_at=stamp, weights={k: v for k, v in weights.items() if v != 0.0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())["session_date"][0]
    assert isinstance(newest, date)
    return newest