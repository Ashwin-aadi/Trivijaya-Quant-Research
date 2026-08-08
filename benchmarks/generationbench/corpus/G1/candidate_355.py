from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression often precedes a breakout or continuation of the trend. "
        "By identifying symbols with reduced price range over a recent period, we can "
        "anticipate potential strong moves in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(history.columns) <= 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        compressed_symbols: list[str] = []

        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol).select(
                pl.col("close"), pl.col("session_date")
            ).sort("session_date")

            if df.height < self._window or len(df.columns) <= 1:
                continue

            open_close_ratio = (df["close"] - df["open"]).abs().max() / (df["high"] - df["low"]).min()
            if open_close_ratio <= 0.2:  # Consider a threshold for range compression
                compressed_symbols.append(symbol)

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest