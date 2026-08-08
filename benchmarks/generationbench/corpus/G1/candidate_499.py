from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression20d(Strategy):
    rationale = (
        "Range compression indicates increased market stability and reduced volatility. "
        "This suggests that the current price is likely to remain within a tight range, making it an opportune time for entry."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        compression_scores = [
            (symbol, float(history[symbol].item()))
            for symbol in symbols
            if abs((history[symbol][-1] - history[symbol][0]) / history[symbol].std()) > 2.0
        ]
        
        if not compression_scores:
            return Signal(information_available_at=stamp, weights={})

        selected_symbols = [symbol for _, (symbol, _) in sorted(compression_scores, key=lambda x: x[1], reverse=True)[:5]]
        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest