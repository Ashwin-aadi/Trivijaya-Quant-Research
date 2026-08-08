from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion exploits the tendency for financial assets to revert to their historical "
        "mean. In a short horizon, extreme deviations from the mean can be expected to correct, "
        "offering trading opportunities."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        recent_close = view.latest_close()
        mean = history.select(
            pl.col("adj_close").mean().alias("mean")
        ).to_dict(full=False)["mean"][0]
        
        deviations = {
            symbol: abs(recent_close[symbol] - mean)
            for symbol in view.symbols
            if symbol in recent_close and symbol in recent_close
        }

        extreme_deviations = [
            (symbol, deviation)
            for symbol, deviation in deviations.items()
            if deviation > self._threshold * mean
        ]

        if not extreme_deviations:
            return Signal(information_available_at=stamp, weights={})

        picks, _ = zip(*extreme_deviations)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_dict(full=False)["session_date"][0]
    assert isinstance(newest, date)
    return newest