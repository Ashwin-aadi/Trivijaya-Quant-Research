from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion suggests that security prices and rental costs of assets tend to "
        "trend towards an average or mean over time. In the context of daily price movements, "
        "prices that deviate significantly from their historical average are likely to revert."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            pl.col("adj_close").mean().alias("mean")
        )
        current_closes = view.closes(lookback=None)
        signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in mean_close.columns or symbol not in current_closes.columns:
                continue
            recent_closes = history.select(pl.col("symbol"), pl.col("adj_close")).filter(
                pl.col("session_date").is_in(history["session_date"].tail(self._window))
            ).sort("session_date")
            if recent_closes.height < self._window:
                continue

            mean = float(mean_close.get_column(symbol).mean())
            current_close = float(current_closes[symbol][0])
            deviation = abs((current_close - mean) / mean)

            if deviation > self._threshold:
                signals[symbol] = 1.0
            else:
                signals[symbol] = 0.0

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest