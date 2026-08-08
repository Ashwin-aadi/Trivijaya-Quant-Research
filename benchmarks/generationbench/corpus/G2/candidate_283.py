from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's price moves less over a period compared to its "
        "previous range. This suggests that the market is consolidating or pausing before potential"
        " breakout or breakdown. Identifying such stocks can help in capturing the subsequent move."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_range = (history["high"] - history["low"]).mean().item()
        compressed_ranges = (
            (history["high"] - history["low"]) / mean_range
        ).to_list()

        compressed_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in compressed_ranges or len(compressed_ranges[symbol]) < self._window:
                continue
            min_compression = min(compressed_ranges[symbol])
            if min_compression <= 0.3:
                compressed_symbols.append(symbol)

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