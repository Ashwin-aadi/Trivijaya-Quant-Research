from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion60d(Strategy):
    rationale = (
        "Recent overbought conditions are often followed by a reversion to more typical price levels. "
        "By focusing on the highest high over the past 20 days and comparing it against a trailing 60-day average, we aim to identify potential reversals."
    )

    def __init__(self, lookback_high: int = 20, lookback_average: int = 60) -> None:
        self._lookback_high = lookback_high
        self._lookback_average = lookback_average

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_average + self._lookback_high)
        if history.is_empty() or history.height < self._lookback_average + self._lookback_high:
            return Signal(information_available_at=stamp, weights={})

        recent_highs = (
            history.with_columns(
                (pl.col("high").rolling_max(window_size=self._lookback_high)).alias("recent_high")
            )
            .sort("session_date", descending=True)
            .select("symbol", "recent_high")
            .group_by("symbol")
            .agg(pl.col("recent_high").max().alias("max_recent_high"))
        )

        trailing_average = (
            history.with_columns(
                (pl.col("high").rolling_mean(window_size=self._lookback_average)).alias("trailing_avg")
            )
            .sort("session_date", descending=True)
            .select("symbol", "trailing_avg")
            .group_by("symbol")
            .agg(pl.col("trailing_avg").max().alias("max_trailing_avg"))
        )

        merged = recent_highs.join(trailing_average, on="symbol", how="inner")

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in merged.columns:
                continue
            recent_max, trailing_max = float(merged[merged["symbol"] == symbol]["max_recent_high"]), float(
                merged[merged["symbol"] == symbol]["max_trailing_avg"]
            )
            if recent_max > trailing_max * 1.05:  # Consider a more aggressive reversion signal
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest