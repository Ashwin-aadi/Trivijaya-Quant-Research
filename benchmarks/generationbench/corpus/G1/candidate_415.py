from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a lack of significant price movement, suggesting that "
        "the market may be consolidating or waiting for a breakout. By identifying symbols with "
        "low range compression, we can potentially find opportunities where the market is likely to "
        "become more volatile."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        range_compression = [
            (high - low) / open_val
            for symbol in symbols
            for high, low, open_val in zip(
                history[f"{symbol}_high"].to_list()[1:],
                history[f"{symbol}_low"].to_list()[1:],
                [float(history[f"{symbol}_open"][0])] * (self._window - 1),
            )
        ]
        if not range_compression:
            return Signal(information_available_at=stamp, weights={})

        min_range_compression = min(range_compression)
        top_symbols = [
            symbol
            for symbol in symbols
            if any(
                (
                    high - low
                    / open_val
                )
                == min_range_compression
                for high, low, open_val in zip(
                    history[f"{symbol}_high"].to_list()[1:],
                    history[f"{symbol}_low"].to_list()[1:],
                    [float(history[f"{symbol}_open"][0])] * (self._window - 1),
                )
            )
        ]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest