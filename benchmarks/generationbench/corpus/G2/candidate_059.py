from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines volume and momentum indicators to identify "
        "overbought and oversold conditions in the NIFTY 100. High volume during a period of strong"
        " upward momentum suggests that institutional investors are active, possibly driving prices higher."
    )

    def __init__(self, window: int = 20, threshold_volume: float = 1.5) -> None:
        self._window = window
        self._threshold_volume = threshold_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        volumes = [float(v) for v in history["volume"].drop_nulls().to_list()]

        breakout_condition = max(closes) >= max(closes[-5:])
        high_volume_condition = any(
            volume > self._threshold_volume * pl.col("volume").mean().item() 
            for volume in volumes
        )

        if not (breakout_condition and high_volume_condition):
            return Signal(information_available_at=stamp, weights={})

        # Find symbols that meet both conditions
        breakout_symbols = [
            symbol for symbol in view.symbols if (
                history[symbol].max() >= max(closes[-5:]) and
                any(
                    float(volume) > self._threshold_volume * pl.col("volume").mean().item()
                    for volume in volumes
                )
            )
        ]

        weight = 1.0 / len(breakout_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in breakout_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest