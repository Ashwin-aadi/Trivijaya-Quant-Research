from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Exploiting the theme of dispersion or range compression in the Indian equity market by "
        "identifying stocks experiencing either high price volatility (dispersion) or limited price "
        "movement (range compression). High dispersion suggests increased uncertainty and potential "
        "for mean reversion, while range compression hints at a stable environment with higher likelihood "
        "of breakout."
    )

    def __init__(self, window: int = 20, threshold_low: float = 1.0, threshold_high: float = 2.5) -> None:
        self._window = window
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        std_devs: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            std_dev = pl.Series(values).std()
            std_devs[symbol] = float(std_dev)

        range_compressed: list[str] = []
        high_dispersion: list[str] = []

        for symbol, std_dev in std_devs.items():
            if std_dev < self._threshold_low:
                range_compressed.append(symbol)
            elif std_dev > self._threshold_high:
                high_dispersion.append(symbol)

        weight_range_compressed = 1.0 / len(range_compressed) if range_compressed else 0.0
        weight_high_dispersion = 1.0 / len(high_dispersion) if high_dispersion else 0.0

        weights: dict[str, float] = {}
        if range_compressed:
            for symbol in range_compressed:
                weights[symbol] = weight_range_compressed
        elif high_dispersion:
            for symbol in high_dispersion:
                weights[symbol] = weight_high_dispersion

        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest