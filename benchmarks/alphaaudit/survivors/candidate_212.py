from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "If a stock makes a significant move in one direction and is accompanied by higher volume, "
        "it often signals continued momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            open_values = [float(v) for v in df["open"].drop_nulls().to_list()]
            close_values = [float(v) for v in df["close"].drop_nulls().to_list()]
            volume_values = [int(v) for v in df["volume"].drop_nulls().to_list()]

            if len(open_values) < self._window or len(close_values) < self._window:
                continue

            last_close = close_values[-1]
            second_last_close = close_values[-2]

            # Calculate the percentage change
            price_move = (last_close - second_last_close) / abs(second_last_close)

            # Ensure we have enough volume to consider it significant
            if volume_values[-1] > sum(volume_values[:-self._window]) * 0.5:
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