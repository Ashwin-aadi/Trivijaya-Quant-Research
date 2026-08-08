from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumAndVolume(Strategy):
    rationale = (
        "Combining momentum and volume signals can capture both trend strength and market "
        "participation. Strong trends are often accompanied by high trading volumes, and this "
        "composite signal aims to identify such instances for potential gains."
    )

    def __init__(self, momentum_window: int = 20, volume_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volume_window = volume_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window)
        if closes.height < self._momentum_window:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_high_momentum_and_volume: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            momentum_score = (values[-1] - values[0]) / sum(values)
            volume_history = view.history(lookback=self._volume_window)
            volume_values = [
                float(v)
                for _, row in volume_history.filter(pl.col("symbol") == symbol).iter_rows()
                if "volume" in row
            ]
            volume_score = sum(volume_values) / len(volume_values)

            if momentum_score > 0.1 and volume_score > 50000:
                symbols_with_high_momentum_and_volume.append(symbol)

        if not symbols_with_high_momentum_and_volume:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_high_momentum_and_volume)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight for s in symbols_with_high_momentum_and_volume
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest