from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to capture trends while scaling trades based on volatility. "
        "Low-volatility periods are associated with persistent price movements, making them "
        "opportune for entering positions in the direction of the trend. The size of these "
        "trades is scaled inversely proportional to the current level of implied volatility."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 19).sort("session_date")
        if history.height < self._window * 2:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        vol = (closes / closes.shift(1) - 1.0).rolling_std(window=self._window)
        sma_long = history["close"].rolling_mean(window=50)
        sma_short = history["close"].rolling_mean(window=200)
        trend_strength_long = (sma_long - sma_short).shift(-1)
        trend_strength_short = (-sma_long + sma_short).abs().shift(-1)

        vol_threshold = 0.05
        low_volatility_mask = vol < vol_threshold

        long_candidates = (
            history.filter(low_volatility_mask)
                   .with_columns((trend_strength_long / vol).alias("scaled_strength"))
                   .sort("scaled_strength", descending=True)
                   .head(self._top_n)["symbol"]
        )

        short_candidates = (
            history.filter(low_volatility_mask)
                   .with_columns((-trend_strength_short / vol).alias("scaled_strength"))
                   .sort("scaled_strength", descending=False)
                   .head(self._top_n)["symbol"]
        )

        long_weights = {s: 1.0 / len(long_candidates) for s in long_candidates}
        short_weights = {s: -1.0 / len(short_candidates) for s in short_candidates}

        weights = {**long_weights, **short_weights}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest