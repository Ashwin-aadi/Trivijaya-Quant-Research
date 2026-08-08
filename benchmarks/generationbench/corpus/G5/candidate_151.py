from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased volatility and potential breakout. "
        "Identifying stocks with reduced price range can signal upcoming momentum."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        symbols = [s for s in view.symbols if s in closes.columns]
        if len(symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        high_low_diffs = {
            symbol: abs(float(history.filter(pl.col("symbol") == symbol)["high"].max())) -
                    abs(float(history.filter(pl.col("symbol") == symbol)["low"].min()))
            for symbol in symbols
        }

        compressed_symbols = [
            symbol for symbol, diff in high_low_diffs.items() if diff <= self._threshold * max(high_low_diffs.values())
        ]
        weights = {symbol: 1.0 / len(compressed_symbols) for symbol in compressed_symbols}

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().date()
    return newest