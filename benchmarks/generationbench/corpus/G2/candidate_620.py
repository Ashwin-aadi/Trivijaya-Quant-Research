from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirm(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "A large volume increase alongside a price move suggests that the move is driven by significant buying or selling pressure."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            latest_close = view.latest_close()[symbol]
            daily_volumes = history[symbol]["volume"].to_list()
            daily_highs = history[symbol]["high"].to_list()
            daily_lows = history[symbol]["low"].to_list()

            if len(daily_volumes) < self._window:
                continue

            price_move_direction = 1.0 if latest_close > daily_highs[-1] else -1.0
            volume_increase = daily_volumes[-1] >= max(daily_volumes)

            if price_move_direction * volume_increase == 1:
                signals[symbol] = 1.0

        return Signal(information_available_at=stamp, weights={k: v for k, v in signals.items() if v > 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest