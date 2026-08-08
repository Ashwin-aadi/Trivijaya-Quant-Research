from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher returns relative to the broader market (NIFTY 100) have historically "
        "outperformed over longer horizons. This can be attributed to their stronger fundamental "
        "strength and better management."
    )

    def __init__(self, window: int = 252, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        nifty_returns = (closes["NIFTY 100"].to_list()[1:] - closes["NIFTY 100"].to_list()[:-1]) / closes["NIFTY 100"].to_list()[:-1]
        symbols_returns = {symbol: (float(closes[symbol].drop_nulls().to_list()[1:]) - float(closes[symbol].drop_nulls().to_list()[:-1])) / float(closes[symbol].drop_nulls().to_list()[:-1]) for symbol in view.symbols if symbol != "NIFTY 100"}

        sorted_symbols = sorted(symbols_returns.items(), key=lambda x: (x[1] - nifty_returns[-1]), reverse=True)[:self._top_n]
        
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s, _ in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest