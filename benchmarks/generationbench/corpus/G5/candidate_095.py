from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy combines short-term and long-term momentum to identify "
        "stocks that are consistently outperforming their peers. Short-term "
        "momentum is captured by the recent price appreciation, while long-term "
        "momentum ensures that the stock has been a reliable performer over time."
    )

    def __init__(self, short_window: int = 10, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._short_window, self._long_window))
        if history.is_empty() or history.height < max(self._short_window, self._long_window):
            return Signal(information_available_at=stamp, weights={})

        short_mom = (history["close"] / history["close"].shift(self._short_window) - 1.0).alias("short_mom")
        long_mom = (history["close"] / history["close"].shift(self._long_window) - 1.0).alias("long_mom")

        momentum_scores = (
            history
            .with_columns(short_mom, long_mom)
            .group_by("symbol")
            .agg(
                pl.col("short_mom").mean().alias("short_mean"),
                pl.col("long_mom").mean().alias("long_mean"),
            )
        )

        if momentum_scores.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks = []
        for symbol in view.symbols:
            short_score = float(momentum_scores.filter(pl.col("symbol") == symbol)["short_mean"].item())
            long_score = float(momentum_scores.filter(pl.col("symbol") == symbol)["long_mean"].item())

            if (short_score > 0.05 and
                long_score > 0.03):  # Thresholds can be adjusted based on backtesting
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest