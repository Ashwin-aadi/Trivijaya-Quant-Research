from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that prices which have deviated significantly from their "
        "historical average are likely to return to it. By identifying such deviations in the "
        "short term (e.g., 5 days), we can construct a trading strategy aimed at capturing this reversal."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.groupby("symbol")
                   .agg((pl.col("adj_close").mean()).alias("mean_close"))
                   .filter(pl.col("session_date") == stamp)
        )

        recent_closes = view.closes(lookback=self._window).select(symbols)

        for symbol in symbols:
            if symbol not in mean_close["symbol"].to_list():
                continue
            recent_values = [float(v) for v in recent_closes[symbol].drop_nulls().to_list()]
            mean_val = float(mean_close.filter(pl.col("symbol") == symbol)["mean_close"][0])
            z_score = (recent_values[-1] - mean_val) / (pl.col("adj_close").std(window=self._window, min_count=1) + 1e-9)

            if abs(z_score) > self._threshold:
                return Signal(
                    information_available_at=stamp,
                    weights={symbol: 1.0 / len(symbols)}
                )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest