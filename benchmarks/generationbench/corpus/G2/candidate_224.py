from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume surges in a direction of recent price movement can indicate strong "
        "buying or selling pressure. A significant increase (or decrease) in volume with the "
        "price moving in the same direction is likely to be followed by further price momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate directional moves and volume changes
        history = (
            history
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("price_return"),
                (pl.col("volume") - pl.col("volume").shift(1)).alias("volume_change")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Identify symbols with significant directional moves and volume changes
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            price_returns = [float(v) for v in history[symbol]["price_return"].to_list()]
            volume_changes = [float(v) for v in history[symbol]["volume_change"].to_list()]

            if len(price_returns) < self._window or len(volume_changes) < self._window:
                continue

            # Check for significant directional move and corresponding volume change
            if (price_returns[-1] > 0 and volume_changes[-1] > 0) or \
               (price_returns[-1] < 0 and volume_changes[-1] < 0):
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
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