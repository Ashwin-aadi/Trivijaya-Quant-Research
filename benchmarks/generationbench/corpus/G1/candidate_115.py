from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating and may be due for a breakout. "
        "High dispersion indicates increased volatility and potential upcoming movement."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_rangedata: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue
            high = max(adj_closes)
            low = min(adj_closes)
            range_width = high - low
            dispersion = sum([abs(price - adj_closes[0]) for price in adj_closes])
            compression_ratio = range_width / (dispersion + 1e-9)  # Avoid division by zero
            symbol_rangedata[symbol] = compression_ratio

        ranked_symbols = sorted(symbol_rangedata, key=symbol_rangedata.get, reverse=True)
        top_symbols = ranked_symbols[:5]
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