from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong buying or selling pressure. "
        "If a stock experiences a significant price move on heavy volume, it is likely to "
        "continue in that direction due to the strength of buyer or seller sentiment."
    )

    def __init__(self, window: int = 10, threshold_volume_ratio: float = 2.0) -> None:
        self._window = window
        self._threshold_volume_ratio = threshold_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Identify volume-confirmed moves
        symbols_with_moves: list[str] = []
        for symbol in view.symbols:
            daily_volumes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["volume"].to_list()]
            if len(daily_volumes) < self._window:
                continue

            # Calculate volume ratio
            latest_volume = daily_volumes[-1]
            mean_volume = sum(daily_volumes[:-1]) / (len(daily_volumes) - 1)
            volume_ratio = latest_volume / mean_volume

            if volume_ratio >= self._threshold_volume_ratio and history.filter(pl.col("symbol") == symbol)["close"].sort(descending=True).head(2)[0] > history.sort("session_date").tail(self._window)["close"].mean():
                symbols_with_moves.append(symbol)

        # Select top N symbols
        symbols_with_moves = sorted(symbols_with_moves, key=lambda x: volume_ratio, reverse=True)[:5]
        
        if not symbols_with_moves:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_moves)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in symbols_with_moves}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest