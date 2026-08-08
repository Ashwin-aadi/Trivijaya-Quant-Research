from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion assumes that asset prices will eventually return to the mean. "
        "In a short horizon, deviations from this mean are temporary but can provide profitable trading opportunities."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(
            pl.col("adj_close").mean().alias("mean")
        ).to_series()[0]
        deviations = (
            (history["adj_close"] - mean_close).abs()
            / history["adj_close"].std(ddof=1)
        )
        
        breakout_symbols = [
            symbol
            for symbol in view.symbols
            if float(deviations[history.height - 1][symbol]) > self._threshold
        ]

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest