from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAverage(Strategy):
    rationale = (
        "Price reversion is a classic technical indicator suggesting that asset prices "
        "tend to revert back to their mean or average level. In this strategy, we calculate the "
        "trailing average of the closing price and use it as a reference point. When the current close "
        "is significantly below this trailing average, it suggests buying opportunities due to reversion."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_close = closes.mean().fill_null(pl.Series([0.0]))
        symbols = [symbol for symbol in view.symbols if symbol in avg_close.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        reversion_scores = {symbol: 0.0 for symbol in symbols}
        for symbol in symbols:
            latest_close = float(view.latest_close()[symbol])
            avg_close_value = float(avg_close[symbol][-1])
            reversion_scores[symbol] = (latest_close - avg_close_value) / avg_close_value

        sorted_symbols = [s for s, _ in sorted(reversion_scores.items(), key=lambda item: item[1], reverse=True)]
        top_n = min(self._window, len(sorted_symbols))
        weight = 1.0 / top_n
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in sorted_symbols[:top_n]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest