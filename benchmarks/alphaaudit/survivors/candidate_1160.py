from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to have stronger momentum. By "
        "identifying stocks that show significant volume on a breakout or breakdown, we can "
        "capitalize on potential sustained price movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol)
            if df.is_empty():
                continue

            closes = df.select(["session_date", "close"])
            volumes = df.select(["session_date", "volume"])

            close_values = [float(v) for v in closes["close"].drop_nulls().to_list()]
            volume_values = [int(v) for v in volumes["volume"].drop_nulls().to_list()]

            if len(close_values) < self._window:
                continue

            latest_close = close_values[-1]
            prev_close = close_values[-2]

            if latest_close > prev_close and max(volume_values[:-1]) < volume_values[-1]:
                picks.append(symbol)
            elif latest_close < prev_close and min(volume_values[:-1]) > volume_values[-1]:
                picks.append(symbol)

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