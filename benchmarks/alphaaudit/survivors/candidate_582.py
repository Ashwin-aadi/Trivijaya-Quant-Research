from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks in the top 30% by strength relative to their peers are selected. "
        "This strategy assumes that stocks outperforming their peers over a certain period may continue "
        "to do so due to fundamental or technical advantages."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_strengths = {}
        for symbol in view.symbols:
            close_prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(close_prices) < self._window:
                continue
            strength = (close_prices[-1] - min(close_prices)) / max(close_prices) - min(close_prices)
            symbol_strengths[symbol] = strength

        sorted_strengths = sorted(symbol_strengths.items(), key=lambda x: x[1], reverse=True)
        top_n_symbols = [symbol for symbol, _ in sorted_strengths[: int(len(view.symbols) * 0.3)]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

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