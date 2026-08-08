from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of strong market sentiment. "
        "A large volume increase on a price move can signal significant buying or selling pressure."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).select(
                "session_date", "open", "close", "volume"
            )
            if hist.height < self._window + 1:
                continue
            recent_close = float(hist.select(pl.last("close")).item())
            prev_close = float(hist.select(pl.first("close")).item())
            vol_change = float(
                (hist.filter(pl.col("session_date") != stamp).select("volume").sum()
                 - hist.filter(pl.col("session_date") == stamp).select("volume").item())
            )
            if (
                recent_close > prev_close and
                vol_change / recent_close >= self._threshold
            ):
                signals[symbol] = 1.0

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest