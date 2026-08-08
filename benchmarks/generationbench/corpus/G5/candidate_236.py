from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume movements in a single direction are often indicative of strong market "
        "sentiment and can lead to continuation of the trend. By identifying such moves, we aim "
        "to capture potentially profitable opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 3)  # Look back a bit more for volume
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        signals: dict[str, float] = {}
        
        for symbol in symbols:
            symbol_history = history.select(["session_date", "close", "volume"])
            recent_closes = symbol_history["close"].to_list()[-self._window:]
            recent_volumes = symbol_history["volume"].to_list()[-self._window:]

            if len(recent_closes) < self._window or any(pl.col("volume").is_null().sum()):
                continue

            direction = [1 if recent_closes[i] > recent_closes[i + 1] else -1 for i in range(self._window - 1)]
            volume_trend = pl.Series(recent_volumes).rolling_sum(window=self._window // 2)

            if any(direction) and all(volume_trend.to_list()):
                signals[symbol] = 1.0 / len(symbols)
        
        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().date()  # Handle the type explicitly
    return newest