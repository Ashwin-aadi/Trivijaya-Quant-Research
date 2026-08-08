from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate strong market sentiment and "
        "can be used to identify potential breakout opportunities or support/resistance levels."
    )

    def __init__(self, window: int = 20, min_volume_threshold: float = 1e6) -> None:
        self._window = window
        self._min_volume_threshold = min_volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).select(
                pl.col("session_date"), "close", "volume"
            ).sort("session_date")

            if df.height < self._window:
                continue

            close_series = [float(v) for v in df["close"].to_list()]
            volume_series = [float(v) for v in df["volume"].to_list()]

            directional_move = max(close_series[-1] - close_series[i] for i in range(self._window))
            if directional_move < 0:
                continue

            recent_volume = volume_series[-1]
            if recent_volume < self._min_volume_threshold:
                continue

            picks.append(symbol)

        picks = picks[:5]  # Limit to top 5 symbols
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
    newest = visible.select(pl.col("session_date").max()).collect()[0][0]
    assert isinstance(newest, date)
    return newest