from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy leverages historical price data to identify trends. By dynamically adjusting "
        "entry points using the 20-day Simple Moving Average (SMA) and volatility, we aim for more "
        "responsive entry signals compared to static thresholds, ensuring timely entries and exits."
    )

    def __init__(self, window: int = 20, threshold: float = 0.02) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or len(history["symbol"].unique()) < 30:
            return Signal(information_available_at=stamp, weights={})

        sma = history.group_by("symbol").agg(
            (pl.col("close").mean().alias("sma"))
        )
        vol = history.with_columns(
            (
                pl.col("close").rolling_std(window=self._window).alias("vol")
            )
        ).group_by("symbol").agg((pl.col("vol") / 20.0).alias("sma_adjusted"))

        merged = sma.join(history, on="symbol", how="inner").join(
            vol, on="symbol", how="inner"
        )

        def entry_signal(row: pl.Series) -> float:
            current_close = row["close"]
            sma = row[f"sma_{self._window}"]
            adjusted_sma = row["sma_adjusted"]
            return 1.0 if current_close > (sma + self._threshold * adjusted_sma) else 0.0

        def exit_signal(row: pl.Series) -> float:
            current_close = row["close"]
            sma = row[f"sma_{self._window}"]
            adjusted_sma = row["sma_adjusted"]
            return -1.0 if current_close < (sma - self._threshold * adjusted_sma) else 0.0

        merged = (
            merged.with_columns(
                pl.Series([entry_signal(row) for _, row in merged.iter_rows()]).alias("signal_entry")
            )
            .with_columns(
                pl.Series([exit_signal(row) for _, row in merged.iter_rows()]).alias("signal_exit")
            )
            .filter((pl.col("signal_entry") == 1.0) | (pl.col("signal_exit") == -1.0))
        )

        if merged.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [row[0] for _, row in merged.iter_rows()]
        weight = 0.02 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest