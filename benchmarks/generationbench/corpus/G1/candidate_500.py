from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "By identifying symbols that show significant price movement alongside increased "
        "volume, we can capture momentum-driven profits."
    )

    def __init__(self, window: int = 20, threshold_volume_increase: float = 1.5) -> None:
        self._window = window
        self._threshold_volume_increase = threshold_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"]]
        volume_signals = []
        price_signals = []

        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol)
            latest_close = float(df.select("adj_close").tail(1).to_list()[0][0])
            prev_close = float(df.select("adj_close").shift(1).tail(20).head(1).to_list()[0][0])

            volume_change = df.filter(pl.col("symbol") == symbol)["volume"].last() / \
                            df.filter(pl.col("symbol") == symbol)["volume"].first()

            if latest_close > prev_close:
                price_signal = 1.0
            elif latest_close < prev_close:
                price_signal = -1.0
            else:
                price_signal = 0.0

            if volume_change >= self._threshold_volume_increase:
                volume_signal = 1.0
            else:
                volume_signal = 0.0

            volume_signals.append(volume_signal)
            price_signals.append(price_signal)

        positive_signals = sum([price * vol for price, vol in zip(price_signals, volume_signals)])
        if not symbols or positive_signals == 0:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        selected_symbols = [symbol for symbol, signal in zip(symbols, volume_signals) if signal > 0]
        weights = {s: weight * (positive_signals / len(selected_symbols)) for s in selected_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest