from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that prices will eventually move back towards an average "
        "level. In a short-term mean reversion strategy, we expect recent extreme movements to be "
        "followed by corrective moves in the opposite direction."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        symbols = view.symbols
        signals: dict[str, float] = {}

        for symbol in symbols:
            if symbol not in closes.columns:
                continue

            # Calculate the mean of close prices over the window
            mean_close = history.select(
                pl.col("adj_close").mean().alias(f"{symbol}_mean")
            )[f"{symbol}_mean"].item()

            # Compute the deviations from the mean for each session
            dev_scores = [
                float((close - mean_close) / mean_close)
                for close in closes[symbol].drop_nulls().to_list()
            ]

            # Identify sessions where prices are far from the mean (potential reversion candidates)
            if any(abs(dev_score) > 1.0 for dev_score in dev_scores):
                signals[symbol] = -1.0 / len(symbols)

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())[0].item()
    assert isinstance(newest, date)
    return newest