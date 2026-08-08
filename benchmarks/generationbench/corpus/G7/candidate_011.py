from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeRatio(Strategy):
    rationale = (
        "By combining daily trading volume with its 20-day simple moving average (SMA), "
        "we can identify stocks experiencing unusual volume spikes. Such spikes may indicate "
        "significant buying or selling pressure and could be a precursor to price movement."
    )

    def __init__(self, lookback_volume: int = 5, window_sma: int = 20) -> None:
        self._lookback_volume = lookback_volume
        self._window_sma = window_sma

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_volume)

        if closes.height < self._lookback_volume + 1:
            return Signal(information_available_at=stamp, weights={})

        volume_ratios: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            sma_volume = sum(values[-self._window_sma:]) / self._window_sma
            ratio = values[-1] / sma_volume
            if ratio > 1.5:  # Example threshold for significant volume spike
                volume_ratios.append((symbol, ratio))

        symbols_with_ratio = [pair[0] for pair in volume_ratios]
        if not symbols_with_ratio:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_ratio)
        return Signal(
            information_available_at=stamp,
            weights={(symbol): weight for symbol in symbols_with_ratio},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest