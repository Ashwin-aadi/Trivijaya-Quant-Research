from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two simple but potentially informative signals: (1) the 20-day"
        " rolling mean of daily returns and (2) the volume anomaly. A high volume on a day when"
        " both signals are strong could indicate a significant market movement, suggesting an "
        "entry point."
    )

    def __init__(self, window: int = 20, volume_threshold: float = 100_000) -> None:
        self._window = window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the 20-day rolling mean of daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("rolling_return"))
        )

        # Filter out symbols with insufficient history
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the latest volume for each symbol
        volumes = view.history(lookback=None).select(["symbol", "volume"])

        # Merge returns and volume data
        signals = (
            history.join(volumes, on="symbol")
            .filter(pl.col("volume") > self._volume_threshold)
        )

        if signals.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Select symbols with the highest rolling return and volume
        picks: list[str] = [
            row["symbol"]
            for _, row in signals.sort("rolling_return", descending=True).rows()
            if row["volume"] > self._volume_threshold
        ][:5]

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