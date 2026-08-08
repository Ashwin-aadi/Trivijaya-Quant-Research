from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong institutional participation and "
        "conviction. High volume on a price move suggests the move is not just noise but likely "
        "due to significant buying or selling pressure, which can be indicative of future gains."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the daily price change and volume
        price_changes = (history["adj_close"] / history["adj_close"].shift(1) - 1).alias("price_change")
        volumes = history["volume"].alias("volume")
        changes_and_volumes = history.with_columns(price_changes, volumes)

        # Filter to only consider days with non-zero price change and sufficient volume
        active_days = (changes_and_volumes.select(pl.all().filter(
            (pl.col("price_change") != 0) & (pl.col("volume") > pl.col("volume").mean() * self._threshold)
        )))

        if active_days.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify the breakout symbols
        breakout_symbols = active_days.select(
            [pl.col("symbol"), price_changes]
        ).sort(price_changes.descending()).head(5)["symbol"].to_list()

        # Assign equal weight to each selected symbol
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest