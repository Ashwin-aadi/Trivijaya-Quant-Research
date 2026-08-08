from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeCharStrategy(Strategy):
    rationale = (
        "This strategy identifies stocks experiencing both strong price momentum and significant trading activity "
        "by analyzing recent price movements and volume trends. It aims to combine these signals to select promising "
        "candidates for investment."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or not history.columns:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue

            price_changes = (
                history.filter(pl.col("symbol") == symbol)
                    .sort("session_date")
                    .select(
                        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("price_change")
                    )
                    .to_series()
                    .to_list()
            )

            if len(price_changes) < self._window:
                continue

            avg_price_change = sum(price_changes[-self._window:]) / self._window
            price_trend = (avg_price_change > 0.01) and all(change >= -0.02 for change in price_changes[-5:])

            volume_changes = (
                history.filter(pl.col("symbol") == symbol)
                    .sort("session_date")
                    .select(
                        pl.col("volume").rolling_sum(window_size=self._window).alias("rolling_volume")
                    )
                    .to_series()
                    .to_list()
            )

            if len(volume_changes) < self._window:
                continue

            avg_volume_change = sum(volume_changes[-self._window:]) / self._window
            volume_trend = (avg_volume_change > 50_000) and all(v >= v - 20000 for v in volume_changes[-5:])

            if price_trend and volume_trend:
                picks.append(symbol)

        weights = {s: 1.0 / len(picks) for s in picks} if picks else {}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest