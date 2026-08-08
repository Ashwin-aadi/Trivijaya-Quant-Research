from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating and may be setting up "
        "for a breakout. This can happen when volatility decreases, suggesting pent-up momentum."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_ranges = {}
        for symbol in view.symbols:
            symbol_data = history.filter(pl.col("symbol") == symbol)
            if symbol_data.is_empty():
                continue
            open_values = [float(v) for v in symbol_data["open"].to_list()]
            close_values = [float(v) for v in symbol_data["close"].to_list()]
            range_value = max(close_values) - min(open_values)
            symbol_ranges[symbol] = range_value

        sorted_ranges = sorted(symbol_ranges.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_ranges[:5]]

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