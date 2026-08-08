from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmDirectionalMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves identify significant shifts in market sentiment "
        "by combining the magnitude of price movement with the volume on that day. High volume "
        "on a strong move is indicative of substantial buying or selling pressure."
    )

    def __init__(self, window: int = 10, threshold_ratio: float = 2.0) -> None:
        self._window = window
        self._threshold_ratio = threshold_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].to_list()) != self._window + 2:
                continue

            open_price = float(history[f"{symbol}"][0])
            close_price = float(history[f"{symbol}"][-1])
            high_price = max(float(high) for high in history[f"{symbol}"].to_list())
            low_price = min(float(low) for low in history[f"{symbol}"].to_list())
            volume_change = float(history[f"{symbol}"][self._window] - history[f"{symbol}"][-1])

            if close_price > open_price and (high_price == close_price or low_price == open_price):
                price_move_direction = "up"
            elif close_price < open_price and (low_price == close_price or high_price == open_price):
                price_move_direction = "down"
            else:
                continue

            if volume_change / max(abs(volume_change), 1) >= self._threshold_ratio:
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