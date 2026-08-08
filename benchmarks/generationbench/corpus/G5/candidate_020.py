from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of strong momentum "
        "and can be used to enter positions that align with the prevailing trend."
    )

    def __init__(self, window: int = 20, volume_threshold: float = 0.7) -> None:
        self._window = window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol)
            if df.height < self._window:
                continue

            close_ratio = (df.select((pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("r"))
                           .select(pl.col("r").mean())
                           .to_series()[0])
            
            volume_ratio = df.filter(
                (pl.col("volume") > df.select(pl.col("volume").median())) &
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0) > 0
            ).height / self._window

            if close_ratio > 0 and volume_ratio >= self._volume_threshold:
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series()[0]
    assert isinstance(newest, date)
    return newest