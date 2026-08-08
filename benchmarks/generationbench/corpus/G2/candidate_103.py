from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that a stock's price action is consolidating within "
        "a narrow range. After such consolidation, the breakout can lead to significant "
        "price movement. Identifying stocks with high range compression can provide "
        "opportunities for profit."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_ratio = (history["high"] - history["low"]) / history["adj_close"]
        mean_range_ratio = range_ratio.mean().item()

        compressed_symbols = []
        for symbol in view.symbols:
            symbol_history = history.select(
                pl.col("session_date"), pl.col(symbol + "_high"), pl.col(symbol + "_low")
            )
            if symbol_history.is_empty():
                continue
            symbol_range_ratio = (
                (symbol_history["symbol_high"] - symbol_history["symbol_low"])
                / symbol_history[symbol].first()
            )
            mean_compression = symbol_range_ratio.mean().item()
            if mean_compression < 0.5 * mean_range_ratio:
                compressed_symbols.append(symbol)

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
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest