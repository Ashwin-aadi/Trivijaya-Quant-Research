from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfDirMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment in a particular "
        "direction. This strategy identifies symbols with significant volume on a close that is "
        "either higher or lower than the previous close, indicating a strong directional move."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_breakouts: dict[str, float] = {}
        for symbol in view.symbols:
            history = view.history().select(["session_date", "close", "volume"]).filter(pl.col("symbol") == symbol)
            if history.height < self._window + 1:
                continue

            latest_close = float(history.select("close").tail(1)["close"][0])
            previous_close = float(history.select("close").tail(2)["close"][-1])

            # Check for volume increase with a directional move
            if (latest_close > previous_close and history["volume"].sum() > 1.5 * history.select("volume").mean()) or \
               (latest_close < previous_close and history["volume"].sum() > 1.5 * history.select("volume").mean()):
                volume_breakouts[symbol] = latest_close

        if not volume_breakouts:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volume_breakouts)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in volume_breakouts.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest