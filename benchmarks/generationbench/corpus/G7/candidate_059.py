from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VWAPDeviationStrategy(Strategy):
    rationale = (
        "The volume-weighted average price (VWAP) deviation from the closing price "
        "provides insight into market sentiment and liquidity. This composite metric can "
        "indicate overbought or oversold conditions when combined with recent price action."
    )

    def __init__(self, window: int = 30, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            volume_series = [float(v) for v in view.history().filter(pl.col("symbol") == symbol)["volume"].to_list()]

            if len(close_series) < self._window or len(volume_series) < self._window:
                continue

            total_volume = sum(volume_series)
            vwap = sum(price * vol for price, vol in zip(close_series, volume_series)) / total_volume
            deviation = (vwap - close_series[-1]) / abs(close_series[-1])

            if len(weights) < self._top_n:
                weights[symbol] = 2.0 / self._top_n

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest