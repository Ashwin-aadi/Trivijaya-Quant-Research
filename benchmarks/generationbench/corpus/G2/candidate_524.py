from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often associated with strong institutional "
        "buying or selling. A significant increase in volume accompanied by a price move can "
        "indicate the beginning of a trend. By identifying such moves, we aim to capture "
        "potential long-term gains."
    )

    def __init__(self, window: int = 20, threshold_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._threshold_volume_ratio = threshold_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * 2:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_prices = [float(v) for v in history[symbol + "_open"].drop_nulls().to_list()]
            close_prices = [float(v) for v in history[symbol + "_close"].drop_nulls().to_list()]
            volumes = [float(v) for v in history[symbol + "_volume"].drop_nulls().to_list()]

            if len(open_prices) < self._window:
                continue

            # Calculate daily price moves
            price_moves = [(close - open_) / open_ for open_, close in zip(open_prices, close_prices)]
            recent_price_move = max(price_moves[-self._window:])
            average_price_move = sum(price_moves) / len(price_moves)

            # Calculate volume changes
            cumulative_volume_change = [
                v1 / v0 if v0 != 0 else 1.0 for v0, v1 in zip(volumes[:-1], volumes[1:])
            ]
            recent_cumulative_volume_change = max(cumulative_volume_change[-self._window:])
            average_cumulative_volume_change = sum(cumulative_volume_change) / len(
                cumulative_volume_change
            )

            # Check if the move is significant and volume confirms it
            if (
                recent_price_move > 2 * average_price_move
                and recent_cumulative_volume_change >= self._threshold_volume_ratio
            ):
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals.keys()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest