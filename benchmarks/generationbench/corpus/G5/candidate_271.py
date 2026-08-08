from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the stock prices are consolidating within a narrow "
        "range. This can be an indication of upcoming breakout or trend reversal, making it a "
        "potentially profitable entry point."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range
        history = history.with_columns(
            (pl.col("high") - pl.col("low")).alias("range")
        )

        # Normalize ranges by the first day's close to compare across symbols
        history = (
            history.with_columns(
                (pl.col("range") / pl.col("close").shift(1)).alias("norm_range")
            )
            .sort("session_date", descending=False)
            .with_column((pl.col("norm_range").rolling_mean(window_size=self._window)).alias("mean_norm_range"))
        )

        # Identify symbols with compressed range
        picks: list[str] = []
        for symbol in view.symbols:
            if f"{symbol}_range" not in history.columns or f"{symbol}_mean_norm_range" not in history.columns:
                continue

            max_range = float(history[f"{symbol}_range"][-1])
            mean_range = float(history[f"{symbol}_mean_norm_range"][-1])

            # Identify potential breakouts or reversals
            if max_range / mean_range <= self._threshold and max_range > 2 * history[f"{symbol}_range"].shift(1):
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest