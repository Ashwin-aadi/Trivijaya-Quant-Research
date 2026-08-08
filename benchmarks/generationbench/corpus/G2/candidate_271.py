from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the price fluctuates more within a given range, "
        "indicating increased volatility and potential for breakout or mean reversion. "
        "By identifying symbols with high range compression, we can capitalize on these "
        "moments of heightened volatility."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_ranges = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            high = max(values)
            low = min(values)
            range_compression = (high - low) / (max(values[-1], 1e-6)) * 100.0
            symbol_ranges.append((symbol, range_compression))

        sorted_ranges = sorted(symbol_ranges, key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_ranges[:5]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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