from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed the broader market over a certain period "
        "are likely to continue outperforming due to momentum effects."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        if closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        close_series: list[float] = []
        for symbol in view.symbols:
            close_series.append(float(closes.get_column(symbol).mean()))
        
        avg_close = sum(close_series) / len(close_series)
        relative_strengths = [(symbol, (close - avg_close) / avg_close) for symbol, close in zip(view.symbols, close_series)]
        sorted_rs = sorted(relative_strengths, key=lambda x: x[1], reverse=True)

        top_n_symbols = [symbol for symbol, _ in sorted_rs[:5]]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest