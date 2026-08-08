from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment and "
        "are often followed by continuation of the trend. This strategy identifies such moves"
        " based on both price change and volume."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        volume = history["volume"]

        price_changes = (closes / closes.shift(1) - 1.0).to_series().abs()
        ranked_price_changes = price_changes.rank(descending=True, method="dense")
        top_price_changes = ranked_price_changes < self._window

        high_volume = volume > volume.quantile(0.75)
        combined_filter = (top_price_changes & high_volume)

        symbols = history.select("symbol")[combined_filter].to_series().to_list()
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(symbols, [weight] * len(symbols))),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest