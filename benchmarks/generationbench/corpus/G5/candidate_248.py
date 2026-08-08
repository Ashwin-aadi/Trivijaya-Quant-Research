from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMoves(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong buying or selling pressure. "
        "We aim to identify stocks that show a significant price move accompanied by increased volume."
    )

    def __init__(self, window: int = 30, threshold_price_change: float = 0.1, min_volume_change_factor: float = 2.0) -> None:
        self._window = window
        self._threshold_price_change = threshold_price_change
        self._min_volume_change_factor = min_volume_change_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            # Calculate price change and volume change
            price_change = float(history.filter(pl.col("symbol") == symbol).select((pl.col("adj_close").last() / pl.col("adj_close").first()) - 1.0).item())
            volume_change_factor = history.filter(pl.col("symbol") == symbol).select(pl.col("volume").sum()).item() / self._window

            # Filter for significant moves with increased volume
            if abs(price_change) >= self._threshold_price_change and volume_change_factor > self._min_volume_change_factor:
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest