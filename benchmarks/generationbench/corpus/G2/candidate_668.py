from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate that a significant number of traders are "
        "committing to the direction of the move. This suggests a strong trend is forming, which "
        "may offer profitable opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_changes: list[float] = []
        price_changes: list[float] = []
        for symbol in view.symbols:
            close_series = history.select(pl.col(symbol).alias("close"))
            volume_series = history.select(pl.col(symbol + "_volume").alias("volume"))

            # Calculate daily changes
            price_changes.extend(
                float(close / prev_close - 1)
                for close, prev_close in zip(
                    close_series["close"].to_list()[1:], close_series["close"].to_list()[:-1]
                )
            )
            volume_changes.extend(
                float(v) for v in volume_series["volume"].to_list()[1:]
            )

        # Filter out non-changing days
        valid_indices = [
            i
            for i, (change, vol_change) in enumerate(zip(price_changes, volume_changes))
            if abs(change) > 0 and abs(vol_change) > 0.2 * max(volume_changes)
        ]

        if not valid_indices:
            return Signal(information_available_at=stamp, weights={})

        # Pick the most recent valid move
        latest_valid_index = valid_indices[-1]

        picks: list[str] = []
        for symbol in view.symbols:
            price_change = float(history.select(pl.col(symbol).alias("close"))["close"].to_list()[latest_valid_index + 1] / history.select(pl.col(symbol).alias("close"))["close"].to_list()[latest_valid_index] - 1)
            if abs(price_change) > max(abs(change) for change in price_changes):
                picks.append(symbol)

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