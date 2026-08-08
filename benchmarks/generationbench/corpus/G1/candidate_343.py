from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the stock prices are trading in a narrower range "
        "than usual. This can be an early sign of an impending breakout or reversal in trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            data = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("session_date"),
                    (pl.col("high") - pl.col("low")).alias("range_width"),
                )
                .sort("session_date")
            )
            if data.height < self._window:
                continue
            last_range = float(data.select(pl.last("range_width")).item())
            mean_range = float(data.agg(pl.col("range_width").mean()).item())
            compression_ratio = last_range / mean_range
            symbol_data[symbol] = (last_range, mean_range, compression_ratio)

        compressed_symbols: list[str] = [
            s for s in symbol_data.keys() if symbol_data[s][2] > 1.5
        ]
        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest