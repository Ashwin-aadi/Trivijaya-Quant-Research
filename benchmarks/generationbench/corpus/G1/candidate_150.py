from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength compared to the broader market "
        "can potentially outperform. This strategy ranks each stock's recent performance and "
        "buys the top performers."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        mean_close = sum(closes) / (self._window * len(view.symbols))
        relative_strengths: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(symbol_closes) < self._window:
                continue
            average_close = sum(symbol_closes) / self._window
            relative_strength = (average_close - mean_close) / mean_close
            relative_strengths[symbol] = relative_strength

        top_n_symbols = sorted(relative_strengths, key=relative_strengths.get, reverse=True)[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest