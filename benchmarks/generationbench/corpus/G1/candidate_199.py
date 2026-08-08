from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of high range compression, market prices move less from day to day. "
        "This suggests a reduction in volatility and increased potential for mean reversion, "
        "favoring a more concentrated portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_values = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_low_diffs = (
                (history[symbol]["high"] - history[symbol]["low"])
                .to_list()
                .max()  # Use max to get the highest daily range
            )
            close_value = float(history[symbol]["close"].last())
            range_compression_values.append((symbol, high_low_diffs / close_value))

        if not range_compression_values:
            return Signal(information_available_at=stamp, weights={})

        sorted_ranges = sorted(range_compression_values, key=lambda x: x[1], reverse=True)
        top_symbols = [x[0] for x in sorted_ranges[:5]]  # Top 5 most compressed ranges
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest