from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals that the market is consolidating and may be setting up "
        "for a breakout. By focusing on stocks with reduced daily price range, we can identify "
        "potential upcoming strong moves."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        ranges = [
            float(
                (history[symbol].item("high").max() - history[symbol].item("low").min())
            )
            for symbol in symbols
        ]
        compressed_sigs = [r / max(ranges) if r > 0 else 0 for r in ranges]
        top_n_symbols = [
            symbols[i] for i in reversed(list(pl.Series(compressed_sigs).top_k(self._window)))
        ]

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest