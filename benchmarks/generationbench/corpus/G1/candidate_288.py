from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategy selects the top-performing stocks in the recent "
        "period based on their price returns. This strategy aims to capture the tendency of "
        "outperforming stocks to continue performing well."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window).sort("session_date", descending=True)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        symbol_closes = {symbol: [float(v) for v in close.to_list()] for symbol, close in view.closes().items()}

        returns = {}
        for symbol, close_prices in symbol_closes.items():
            if len(close_prices) < self._window:
                continue
            returns[symbol] = (close_prices[-1] / close_prices[0]) - 1.0

        top_symbols = sorted(returns.keys(), key=lambda x: returns[x], reverse=True)[:self._top_n]
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