from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment. When a stock "
        "moves significantly in the direction of its trend and is accompanied by increased "
        "volume, it often indicates a stronger continuation of that trend. This can be used "
        "to identify high-probability trading opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)
        if history.is_empty() or history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        def volume_confirmed_move(symbol: str) -> bool:
            symbol_history = history.filter(pl.col("symbol") == symbol)
            recent_close = float(view.latest_close()[symbol])
            recent_high = float(symbol_history.select(pl.max("high")).item())
            recent_low = float(symbol_history.select(pl.min("low")).item())
            recent_open = float(symbol_history.select(pl.first("open")).item())

            volume_change = (
                float(history.filter(pl.col("symbol") == symbol)
                      .select((pl.col("volume").sum()).alias("vol_sum"))
                      .item())
                - history.filter(pl.col("symbol") == symbol)
                          .select((pl.col("volume").shift(1).sum()).alias("prev_vol_sum"))
                          .item()
            )

            if recent_close > recent_open:
                return (recent_close >= recent_high) and volume_change > 0
            else:
                return (recent_close <= recent_low) and volume_change > 0

        picks = [symbol for symbol in view.symbols if volume_confirmed_move(symbol)]
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