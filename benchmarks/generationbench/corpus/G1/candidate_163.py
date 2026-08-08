from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price is consolidating, which can precede "
        "a breakout or a reversal. By identifying symbols with reduced volatility, we aim to "
        "capture potential breakout opportunities."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_low_diff = float(history[symbol]["high"].max()) - float(
                history[symbol]["low"].min()
            )
            close_open_ratio = (
                float(history[symbol]["close"][-1])
                / float(history[symbol]["open"][0])
            )

            # Filter out symbols with high range or close to opening
            if 0.5 <= close_open_ratio < 1.5 and high_low_diff < history.height * 0.2:
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