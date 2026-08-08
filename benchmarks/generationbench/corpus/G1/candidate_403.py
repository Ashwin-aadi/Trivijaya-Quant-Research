from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased market consolidation and reduced volatility. "
        "This can often precede a breakout or trend reversal, making it a potential entry point."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_range = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_data = [float(v) for v in history[symbol].to_list()]
            high_low_ratio = max(close_data) / min(close_data)
            symbol_range[symbol] = high_low_ratio

        sorted_symbols = sorted(symbol_range.items(), key=lambda x: x[1], reverse=True)
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = [s for s, _ in sorted_symbols[:5]]
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